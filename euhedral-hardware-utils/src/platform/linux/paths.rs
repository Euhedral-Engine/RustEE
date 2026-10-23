//! Locates the calling process's cgroup directories (upstream `LinuxPaths`).
//!
//! Everything is resolved under one filesystem root, `/` in production and a fixture directory in
//! tests, from `<root>/proc/self/mountinfo` and `<root>/proc/self/cgroup`.

use super::fs::read_bounded;
use super::parse::unescape_mount_path;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

/// Which cgroup hierarchies the process is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CgroupMode {
    V2,
    V1,
    /// Both: v1 controllers and the unified hierarchy.
    Hybrid,
    /// No cgroup information; host-wide sources only.
    BareHost,
}

#[derive(Clone, Debug)]
struct Mount {
    /// The hierarchy path mounted, as mountinfo's fourth field.
    root: String,
    /// Where it is mounted, under the filesystem root.
    point: PathBuf,
}

#[derive(Clone, Debug)]
pub(crate) struct LinuxPaths {
    root: PathBuf,
    mode: CgroupMode,
    v2: Option<PathBuf>,
    v1: HashMap<String, PathBuf>,
}

/// `base` joined with an absolute or relative path whose components never climb above `base`.
fn join_under(base: &Path, path: &str) -> Option<PathBuf> {
    let mut joined = base.to_path_buf();
    for component in Path::new(path).components() {
        match component {
            Component::Normal(part) => joined.push(part),
            Component::RootDir | Component::CurDir => {}
            Component::ParentDir | Component::Prefix(_) => return None,
        }
    }
    Some(joined)
}

/// The directory of hierarchy path `path` inside `mount`, if the mount contains it.
fn locate(mount: &Mount, path: &str) -> Option<PathBuf> {
    let root = mount.root.trim_end_matches('/');
    let relative = path.strip_prefix(root)?;
    if !relative.is_empty() && !relative.starts_with('/') {
        return None;
    }
    join_under(&mount.point, relative)
}

impl LinuxPaths {
    /// Reads the cgroup layout under `root`. Missing or malformed files leave the affected hierarchy
    /// unknown; a directory that does not exist is never used.
    pub(crate) fn discover(root: &Path) -> Self {
        let (v2_mounts, v1_mounts) = Self::mounts(root);
        let membership = read_bounded(&root.join("proc/self/cgroup")).unwrap_or_default();
        let mut v2 = None;
        let mut v1 = HashMap::new();
        for line in membership.lines() {
            let mut parts = line.splitn(3, ':');
            let (Some(hierarchy), Some(controllers), Some(path)) =
                (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            if hierarchy == "0" && controllers.is_empty() {
                let fallback = Mount {
                    root: "/".into(),
                    point: root.join("sys/fs/cgroup"),
                };
                v2 = v2_mounts
                    .iter()
                    .chain(v2_mounts.is_empty().then_some(&fallback))
                    .find_map(|mount| locate(mount, path).filter(|dir| dir.is_dir()));
            } else {
                for controller in controllers
                    .split(',')
                    .filter(|c| !c.is_empty() && !c.starts_with("name="))
                {
                    let dir = v1_mounts
                        .get(controller)
                        .and_then(|mount| locate(mount, path))
                        .filter(|dir| dir.is_dir());
                    if let Some(dir) = dir {
                        v1.insert(controller.to_owned(), dir);
                    }
                }
            }
        }
        let mode = match (v2.is_some(), !v1.is_empty()) {
            (true, true) => CgroupMode::Hybrid,
            (true, false) => CgroupMode::V2,
            (false, true) => CgroupMode::V1,
            (false, false) => CgroupMode::BareHost,
        };
        Self {
            root: root.to_path_buf(),
            mode,
            v2,
            v1,
        }
    }

    /// The cgroup2 mounts and the v1 mount of each controller.
    fn mounts(root: &Path) -> (Vec<Mount>, HashMap<String, Mount>) {
        let mut v2 = Vec::new();
        let mut v1 = HashMap::new();
        let Ok(mountinfo) = read_bounded(&root.join("proc/self/mountinfo")) else {
            return (v2, v1);
        };
        for line in mountinfo.lines() {
            let fields: Vec<&str> = line.split(' ').collect();
            let Some(separator) = fields
                .iter()
                .skip(6)
                .position(|&field| field == "-")
                .map(|i| i + 6)
            else {
                continue;
            };
            let (Some(fs_type), Some(&mount_root), Some(&point)) =
                (fields.get(separator + 1), fields.get(3), fields.get(4))
            else {
                continue;
            };
            let Some(point) = join_under(root, &unescape_mount_path(point)) else {
                continue;
            };
            let mount = Mount {
                root: unescape_mount_path(mount_root),
                point,
            };
            match *fs_type {
                "cgroup2" => v2.push(mount),
                "cgroup" => {
                    let super_options = fields.get(separator + 3).copied().unwrap_or("");
                    for option in super_options.split(',') {
                        if !matches!(option, "" | "rw" | "ro") && !option.contains('=') {
                            v1.entry(option.to_owned()).or_insert_with(|| mount.clone());
                        }
                    }
                }
                _ => {}
            }
        }
        (v2, v1)
    }

    pub(crate) fn mode(&self) -> CgroupMode {
        self.mode
    }

    /// `<root>/proc/<relative>`.
    pub(crate) fn proc(&self, relative: &str) -> PathBuf {
        self.root.join("proc").join(relative)
    }

    /// `<root>/sys/<relative>`.
    pub(crate) fn sys(&self, relative: &str) -> PathBuf {
        self.root.join("sys").join(relative)
    }

    /// A file of the process's cgroup v2 directory, if it exists.
    pub(crate) fn v2(&self, file: &str) -> Option<PathBuf> {
        self.v2
            .as_ref()
            .map(|dir| dir.join(file))
            .filter(|path| path.exists())
    }

    /// A file of the process's directory for a v1 controller, if it exists.
    pub(crate) fn v1(&self, controller: &str, file: &str) -> Option<PathBuf> {
        self.v1
            .get(controller)
            .map(|dir| dir.join(file))
            .filter(|path| path.exists())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::testing::{TempDir, write};

    const V2_MOUNTINFO: &str = "29 23 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw,nsdelegate\n\
                                22 1 8:1 / / rw,relatime - ext4 /dev/sda1 rw\n";

    #[test]
    fn unified_hierarchy() {
        let dir = TempDir::new("paths-v2");
        write(dir.path(), "proc/self/mountinfo", V2_MOUNTINFO);
        write(dir.path(), "proc/self/cgroup", "0::/user.slice/app.scope\n");
        write(
            dir.path(),
            "sys/fs/cgroup/user.slice/app.scope/cpu.max",
            "max 100000\n",
        );
        write(dir.path(), "sys/fs/cgroup/cpu.max", "host\n");
        let paths = LinuxPaths::discover(dir.path());
        assert_eq!(paths.mode(), CgroupMode::V2);
        assert_eq!(
            paths.v2("cpu.max"),
            Some(
                dir.path()
                    .join("sys/fs/cgroup/user.slice/app.scope/cpu.max")
            )
        );
        // A file missing from the process's cgroup is never taken from the host's root cgroup.
        write(dir.path(), "sys/fs/cgroup/memory.max", "1\n");
        assert_eq!(paths.v2("memory.max"), None);
    }

    #[test]
    fn a_namespaced_mount_root_is_stripped() {
        let dir = TempDir::new("paths-ns");
        write(
            dir.path(),
            "proc/self/mountinfo",
            "40 30 0:30 /kubepods/pod1 /sys/fs/cgroup ro - cgroup2 cgroup rw\n",
        );
        write(dir.path(), "proc/self/cgroup", "0::/kubepods/pod1/ctr\n");
        write(dir.path(), "sys/fs/cgroup/ctr/cpu.stat", "usage_usec 1\n");
        let paths = LinuxPaths::discover(dir.path());
        assert_eq!(
            paths.v2("cpu.stat"),
            Some(dir.path().join("sys/fs/cgroup/ctr/cpu.stat"))
        );
    }

    #[test]
    fn v1_controllers_and_escaped_mount_points() {
        let dir = TempDir::new("paths-v1");
        write(
            dir.path(),
            "proc/self/mountinfo",
            "31 25 0:27 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid - cgroup cgroup rw,cpu,cpuacct\n\
             32 25 0:28 / /sys/fs/cgroup/mem\\040ory rw - cgroup cgroup rw,memory\n\
             33 25 0:29 / /sys/fs/cgroup/systemd rw - cgroup cgroup rw,xattr,name=systemd\n",
        );
        write(
            dir.path(),
            "proc/self/cgroup",
            "4:cpu,cpuacct:/docker/abc\n3:memory:/docker/abc\n1:name=systemd:/docker/abc\n",
        );
        write(
            dir.path(),
            "sys/fs/cgroup/cpu,cpuacct/docker/abc/cpu.cfs_quota_us",
            "50000\n",
        );
        write(
            dir.path(),
            "sys/fs/cgroup/mem ory/docker/abc/memory.limit_in_bytes",
            "1024\n",
        );
        let paths = LinuxPaths::discover(dir.path());
        assert_eq!(paths.mode(), CgroupMode::V1);
        assert!(paths.v1("cpu", "cpu.cfs_quota_us").is_some());
        assert!(paths.v1("cpuacct", "cpu.cfs_quota_us").is_some());
        assert!(paths.v1("memory", "memory.limit_in_bytes").is_some());
        assert_eq!(paths.v1("cpuset", "cpuset.cpus"), None);
    }

    #[test]
    fn hybrid_and_bare_host() {
        let dir = TempDir::new("paths-hybrid");
        write(
            dir.path(),
            "proc/self/mountinfo",
            "31 25 0:27 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n\
             30 25 0:26 / /sys/fs/cgroup/unified rw - cgroup2 cgroup2 rw\n",
        );
        write(dir.path(), "proc/self/cgroup", "3:memory:/a\n0::/a\n");
        write(dir.path(), "sys/fs/cgroup/memory/a/memory.stat", "");
        write(dir.path(), "sys/fs/cgroup/unified/a/cgroup.procs", "");
        assert_eq!(LinuxPaths::discover(dir.path()).mode(), CgroupMode::Hybrid);

        let bare = TempDir::new("paths-bare");
        let paths = LinuxPaths::discover(bare.path());
        assert_eq!(paths.mode(), CgroupMode::BareHost);
        assert_eq!(paths.proc("stat"), bare.path().join("proc/stat"));
    }

    #[test]
    fn hierarchy_paths_cannot_escape_the_mount() {
        let dir = TempDir::new("paths-escape");
        write(dir.path(), "proc/self/mountinfo", V2_MOUNTINFO);
        write(dir.path(), "proc/self/cgroup", "0::/../../etc\n");
        assert_eq!(
            LinuxPaths::discover(dir.path()).mode(),
            CgroupMode::BareHost
        );
    }
}
