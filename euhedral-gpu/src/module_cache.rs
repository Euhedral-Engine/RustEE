//! A cache of compiled modules, keyed by a hash of everything that went into them.
//!
//! The key covers the source, each option in order, the in-memory headers, the include paths (as strings), the target
//! and the compiler's version and build. The files under the include paths are not hashed: when they change, change the
//! options (a `-D` of a revision number, say) or the headers, or use another cache directory. Entries are written to a temporary file and
//! renamed into place, so a reader sees a whole file or none, even with two processes writing one key.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use euhedral_hashing::hasher_api::hash_bytes_with_seed;

use crate::CompileOptions;

/// 128 bits of a compilation's inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey(pub u64, pub u64);

/// What a compilation depends on.
#[derive(Clone, Copy, Debug)]
pub struct KeyMaterial<'a> {
    pub source: &'a str,
    pub options: &'a CompileOptions,
    /// The target the compiler is asked for, for example `sm_120`.
    pub target: &'a str,
    pub compiler_version: (i32, i32),
    /// Names the compiler's build beyond its version (its patch level): see `Nvrtc::build`.
    pub compiler_build: &'a str,
}

fn field(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// The key of `material`.
pub fn cache_key(material: &KeyMaterial<'_>) -> CacheKey {
    let mut bytes = b"euhedral-gpu-module-v1\0".to_vec();
    field(&mut bytes, material.source.as_bytes());
    bytes.extend_from_slice(&(material.options.options.len() as u64).to_le_bytes());
    for option in &material.options.options {
        field(&mut bytes, option.as_bytes());
    }
    bytes.extend_from_slice(&(material.options.headers.len() as u64).to_le_bytes());
    for (name, contents) in &material.options.headers {
        field(&mut bytes, name.as_bytes());
        field(&mut bytes, contents.as_bytes());
    }
    bytes.extend_from_slice(&(material.options.include_paths.len() as u64).to_le_bytes());
    for path in &material.options.include_paths {
        field(&mut bytes, path.to_string_lossy().as_bytes());
    }
    field(&mut bytes, material.target.as_bytes());
    bytes.extend_from_slice(&material.compiler_version.0.to_le_bytes());
    bytes.extend_from_slice(&material.compiler_version.1.to_le_bytes());
    field(&mut bytes, material.compiler_build.as_bytes());
    CacheKey(
        hash_bytes_with_seed(&bytes, 0x6d6f_6475_6c65_0001),
        hash_bytes_with_seed(&bytes, 0x6d6f_6475_6c65_0002),
    )
}

/// A directory of compiled modules. Reading and writing it blocks: it is used where modules are loaded, which is a
/// build-time call.
#[derive(Clone, Debug)]
pub struct ModuleCache {
    dir: PathBuf,
}

static STORES: AtomicU64 = AtomicU64::new(0);

#[allow(
    clippy::disallowed_methods,
    reason = "module loading is a build-time call; it blocks by nature (compile, driver load)"
)]
impl ModuleCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The file of `key` with extension `ext`.
    pub fn path(&self, key: CacheKey, ext: &str) -> PathBuf {
        self.dir.join(format!("{:016x}{:016x}.{ext}", key.0, key.1))
    }

    /// The entry, or `None` when there is none (or it cannot be read).
    pub fn load(&self, key: CacheKey, ext: &str) -> Option<Vec<u8>> {
        std::fs::read(self.path(key, ext)).ok()
    }

    /// Writes the entry to a temporary file and renames it over the final name.
    pub fn store(&self, key: CacheKey, ext: &str, bytes: &[u8]) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.path(key, ext);
        let temporary = path.with_extension(format!(
            "{ext}.tmp-{}-{}",
            std::process::id(),
            STORES.fetch_add(1, Ordering::Relaxed)
        ));
        {
            // The bytes reach the disk before the name does, so a crash leaves a whole entry or none.
            let mut file = std::fs::File::create(&temporary)?;
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        std::fs::rename(&temporary, &path).inspect_err(|_| {
            let _ = std::fs::remove_file(&temporary);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory in the build's target directory (never the system's temporary directory).
    #[allow(clippy::disallowed_methods, reason = "a test fixture")]
    fn scratch(name: &str) -> PathBuf {
        let exe = std::env::current_exe().unwrap();
        let dir = exe
            .parent()
            .unwrap()
            .join("module-cache-tests")
            .join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Everything a key covers, to edit one input at a time.
    #[derive(Clone)]
    struct Inputs {
        options: CompileOptions,
        source: String,
        target: String,
        version: (i32, i32),
        build: String,
    }

    fn inputs() -> Inputs {
        Inputs {
            options: CompileOptions {
                options: vec!["--std=c++17".into(), "-O3".into()],
                headers: vec![("a.h".into(), "#define A 1".into())],
                include_paths: vec![PathBuf::from("/inc")],
                ..CompileOptions::default()
            },
            source: "kernel".into(),
            target: "sm_120".into(),
            version: (13, 1),
            build: "libnvrtc.so.13.1.80:1000:1".into(),
        }
    }

    fn key(i: &Inputs) -> CacheKey {
        cache_key(&KeyMaterial {
            source: &i.source,
            options: &i.options,
            target: &i.target,
            compiler_version: i.version,
            compiler_build: &i.build,
        })
    }

    #[test]
    fn the_key_changes_with_each_input() {
        let base = key(&inputs());
        assert_eq!(base, key(&inputs()));
        let edits: [fn(&mut Inputs); 9] = [
            |i| i.source.push(' '),
            |i| i.options.options[0] = "--std=c++20".into(),
            |i| i.options.options.reverse(),
            |i| i.options.headers[0].1 = "#define A 2".into(),
            |i| i.options.headers[0].0 = "b.h".into(),
            |i| i.options.include_paths[0] = PathBuf::from("/other"),
            |i| i.target = "sm_90".into(),
            |i| i.version = (13, 2),
            |i| i.build = "libnvrtc.so.13.1.115:1000:1".into(),
        ];
        for (n, edit) in edits.iter().enumerate() {
            let mut edited = inputs();
            edit(&mut edited);
            assert_ne!(key(&edited), base, "edit {n}");
        }
    }

    #[test]
    #[allow(clippy::disallowed_methods, reason = "a test fixture")]
    fn stores_are_atomic_renames() {
        let dir = scratch("atomic");
        let cache = ModuleCache::new(&dir);
        let k = CacheKey(1, 2);
        cache.store(k, "cubin", b"first").unwrap();
        let names = |dir: &PathBuf| -> Vec<String> {
            std::fs::read_dir(dir)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect()
        };
        assert_eq!(names(&dir), [format!("{:016x}{:016x}.cubin", 1, 2)]);
        assert_eq!(cache.load(k, "cubin").as_deref(), Some(&b"first"[..]));
        cache.store(k, "cubin", b"second, longer").unwrap();
        assert_eq!(
            cache.load(k, "cubin").as_deref(),
            Some(&b"second, longer"[..])
        );
        assert_eq!(names(&dir).len(), 1, "no temporary file is left");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_entry_loads_as_none() {
        let cache = ModuleCache::new(scratch("missing"));
        assert_eq!(cache.load(CacheKey(3, 4), "ptx"), None);
    }
}
