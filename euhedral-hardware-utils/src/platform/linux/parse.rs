//! Total parsers for the `/proc`, sysfs and cgroup text formats (upstream `LinuxProcParsers`). Every
//! function returns `None` for malformed, truncated or overflowing input and never panics.

use crate::topology::source::MAX_LOGICAL_CPU_ID;
use crate::util::CpuSet;

/// A plain non-negative decimal.
pub(crate) fn decimal(text: &str) -> Option<u64> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// The whole file, surrounding whitespace ignored, as one decimal.
pub(crate) fn single(content: &str) -> Option<u64> {
    decimal(content.trim())
}

/// Linux cpulist syntax such as `0-3,6,8-11`, whitespace around items allowed. `None` for empty or
/// malformed input, reversed ranges or ids above [`MAX_LOGICAL_CPU_ID`].
pub(crate) fn cpu_list(text: &str) -> Option<CpuSet> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let id = |item: &str| {
        let item = item.trim();
        (item.len() <= 7)
            .then(|| decimal(item))
            .flatten()
            .filter(|&id| id <= MAX_LOGICAL_CPU_ID as u64)
    };
    let mut cpus = Vec::new();
    for item in text.split(',') {
        let (low, high) = match item.split_once('-') {
            Some((low, high)) => (id(low)?, id(high)?),
            None => {
                let single = id(item)?;
                (single, single)
            }
        };
        if high < low {
            return None;
        }
        cpus.push((low as usize, high as usize + 1));
    }
    let mut words = Vec::new();
    for (low, high) in cpus {
        for cpu in low..high {
            crate::util::cpu_set::set_bit(&mut words, cpu);
        }
    }
    Some(CpuSet::from_words(&words))
}

/// The value of a `key value` line (`cpu.stat`, `memory.stat`, `/proc/vmstat`).
pub(crate) fn keyed(content: &str, key: &str) -> Option<u64> {
    keyed_line(content, key).and_then(|rest| decimal(rest.split_whitespace().next()?))
}

/// The rest of the first line whose first token is `key`.
pub(crate) fn keyed_line<'a>(content: &'a str, key: &str) -> Option<&'a str> {
    content.lines().find_map(|line| {
        let rest = line.strip_prefix(key)?;
        rest.starts_with(char::is_whitespace).then_some(rest)
    })
}

/// A `/proc/meminfo` value, in bytes.
pub(crate) fn meminfo_bytes(content: &str, key: &str) -> Option<u64> {
    content.lines().find_map(|line| {
        let rest = line.strip_prefix(key)?.strip_prefix(':')?;
        let mut tokens = rest.split_whitespace();
        let value = decimal(tokens.next()?)?;
        match tokens.next() {
            Some("kB") => value.checked_mul(1024),
            None => Some(value),
            Some(_) => None,
        }
    })
}

/// The `total=` microsecond counter of a PSI file's `some` line, in nanoseconds.
pub(crate) fn psi_some_total_ns(content: &str) -> Option<u64> {
    let line = content.lines().find(|line| line.starts_with("some "))?;
    let total = line
        .split_whitespace()
        .find_map(|token| token.strip_prefix("total="))?;
    decimal(total)?.checked_mul(1_000)
}

/// A cgroup v2 `cpu.max`: the quota in microseconds (`None` for `max`) and the period.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CpuMax {
    pub(crate) quota_us: Option<u64>,
    pub(crate) period_us: u64,
}

/// `None` when malformed, or when the period or an explicit quota is zero.
pub(crate) fn cpu_max(content: &str) -> Option<CpuMax> {
    let mut tokens = content.split_whitespace();
    let quota = tokens.next()?;
    let period_us = decimal(tokens.next()?).filter(|&period| period > 0)?;
    if tokens.next().is_some() {
        return None;
    }
    let quota_us = match quota {
        "max" => None,
        quota => Some(decimal(quota).filter(|&quota| quota > 0)?),
    };
    Some(CpuMax {
        quota_us,
        period_us,
    })
}

/// A `/proc/stat` per-CPU line in jiffies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StatLine {
    pub(crate) user: u64,
    pub(crate) nice: u64,
    pub(crate) system: u64,
    pub(crate) idle: u64,
    pub(crate) iowait: u64,
    pub(crate) irq: u64,
    pub(crate) softirq: u64,
    pub(crate) steal: u64,
}

impl StatLine {
    /// Time spent running work, excluding idle, iowait and steal.
    pub(crate) fn busy(&self) -> u64 {
        [self.user, self.nice, self.system, self.irq, self.softirq]
            .iter()
            .fold(0u64, |sum, &v| sum.saturating_add(v))
    }
}

/// `cpu` (aggregate, id `None`) or `cpuN` lines of `/proc/stat`. At least four counters are required;
/// later ones default to zero on old kernels.
pub(crate) fn stat_line(line: &str) -> Option<(Option<usize>, StatLine)> {
    let mut tokens = line.split_whitespace();
    let label = tokens.next()?.strip_prefix("cpu")?;
    let id = match label {
        "" => None,
        id => Some(
            usize::try_from(decimal(id)?)
                .ok()
                .filter(|&id| id <= MAX_LOGICAL_CPU_ID)?,
        ),
    };
    let mut fields = [0u64; 8];
    let mut count = 0;
    for (slot, token) in fields.iter_mut().zip(tokens.by_ref()) {
        *slot = decimal(token)?;
        count += 1;
    }
    if count < 4 {
        return None;
    }
    let [user, nice, system, idle, iowait, irq, softirq, steal] = fields;
    Some((
        id,
        StatLine {
            user,
            nice,
            system,
            idle,
            iowait,
            irq,
            softirq,
            steal,
        },
    ))
}

/// `cpuN` lines of `/proc/schedstat`: the id and the cumulative run-queue wait (`run_delay`), ns.
pub(crate) fn schedstat_line(line: &str) -> Option<(usize, u64)> {
    let mut tokens = line.split_whitespace();
    let id = usize::try_from(decimal(tokens.next()?.strip_prefix("cpu")?)?).ok()?;
    // yld_count, (legacy), schedule, schedule_idle, ttwu, ttwu_local, rq_cpu_time, run_delay, ...
    let run_delay = decimal(tokens.nth(7)?)?;
    Some((id, run_delay))
}

/// One `/proc/diskstats` record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DiskStat {
    pub(crate) name: String,
    pub(crate) reads: u64,
    pub(crate) sectors_read: u64,
    pub(crate) read_ms: u64,
    pub(crate) writes: u64,
    pub(crate) sectors_written: u64,
    pub(crate) write_ms: u64,
    pub(crate) in_flight: u64,
}

/// `None` for a record with fewer than the 14 classic fields or a malformed counter.
pub(crate) fn diskstats_line(line: &str) -> Option<DiskStat> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < 14 {
        return None;
    }
    let field = |index: usize| decimal(tokens[index]);
    decimal(tokens[0])?;
    decimal(tokens[1])?;
    Some(DiskStat {
        name: tokens[2].to_owned(),
        reads: field(3)?,
        sectors_read: field(5)?,
        read_ms: field(6)?,
        writes: field(7)?,
        sectors_written: field(9)?,
        write_ms: field(10)?,
        in_flight: field(11)?,
    })
}

/// The total of `rbytes` and `wbytes` over every device of a cgroup v2 `io.stat`. A file with no
/// device lines reads 0 (a new cgroup); a malformed counter is `None`.
pub(crate) fn io_stat_bytes(content: &str) -> Option<u64> {
    let mut total = 0u64;
    for line in content.lines() {
        for token in line.split_whitespace().skip(1) {
            if let Some(value) = token
                .strip_prefix("rbytes=")
                .or_else(|| token.strip_prefix("wbytes="))
            {
                total = total.saturating_add(decimal(value)?);
            }
        }
    }
    Some(total)
}

/// A sysfs cache size such as `32K`, `1M`, `2G` or `4096`.
pub(crate) fn size_bytes(text: &str) -> Option<u64> {
    let text = text.trim();
    let (digits, scale) = match text.as_bytes().last()? {
        b'K' | b'k' => (&text[..text.len() - 1], 1u64 << 10),
        b'M' | b'm' => (&text[..text.len() - 1], 1 << 20),
        b'G' | b'g' => (&text[..text.len() - 1], 1 << 30),
        _ => (text, 1),
    };
    decimal(digits)?.checked_mul(scale)
}

/// Decodes the octal escapes (`\040` for a space) mountinfo uses in paths.
pub(crate) fn unescape_mount_path(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\'
            && i + 3 < bytes.len()
            && bytes[i + 1..i + 4]
                .iter()
                .all(|b| (b'0'..=b'7').contains(b))
        {
            let value = (bytes[i + 1] - b'0') as u32 * 64
                + (bytes[i + 2] - b'0') as u32 * 8
                + (bytes[i + 3] - b'0') as u32;
            if let Ok(byte) = u8::try_from(value) {
                out.push(byte);
                i += 4;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(ids: &[usize]) -> CpuSet {
        ids.iter().copied().collect()
    }

    #[test]
    fn cpu_lists() {
        assert_eq!(cpu_list("0-3,6,8-9"), Some(set(&[0, 1, 2, 3, 6, 8, 9])));
        assert_eq!(cpu_list(" 4\n"), Some(set(&[4])));
        assert_eq!(cpu_list(" 0 - 1 , 5 \n"), Some(set(&[0, 1, 5])));
        assert_eq!(cpu_list("1048575"), Some(set(&[1_048_575])));
        for bad in [
            "",
            "  \n",
            "3-1",
            "a",
            "1-b",
            "1,,2",
            "1,",
            ",1",
            "-1",
            "1-",
            "1-2-3",
            "0x3",
            "+1",
            "99999999999",
            "1048576",
        ] {
            assert_eq!(cpu_list(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn cpu_max_quota_and_period() {
        assert_eq!(
            cpu_max("max 100000\n"),
            Some(CpuMax {
                quota_us: None,
                period_us: 100_000
            })
        );
        assert_eq!(
            cpu_max("50000 100000"),
            Some(CpuMax {
                quota_us: Some(50_000),
                period_us: 100_000
            })
        );
        for bad in [
            "",
            "max",
            "max 0",
            "0 100000",
            "-1 100000",
            "x 100000",
            "1 2 3",
            "50000 abc",
        ] {
            assert_eq!(cpu_max(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn stat_lines() {
        let (id, line) = stat_line("cpu3 10 1 5 100 7 2 3 4 0 0").unwrap();
        assert_eq!(id, Some(3));
        assert_eq!((line.iowait, line.steal, line.busy()), (7, 4, 21));
        let (id, old) = stat_line("cpu 10 1 5 100").unwrap();
        assert_eq!((id, old.steal), (None, 0));
        for bad in [
            "cpu3 1 2 3",
            "cpux 1 2 3 4",
            "cpu3 1 2 x 4",
            "intr 1 2 3 4",
            "cpu-1 1 2 3 4",
        ] {
            assert_eq!(stat_line(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn schedstat_lines() {
        assert_eq!(
            schedstat_line("cpu2 0 0 50 10 20 5 900000 123456 77"),
            Some((2, 123_456))
        );
        assert_eq!(schedstat_line("cpu2 0 0 50"), None);
        assert_eq!(schedstat_line("domain0 3 1 2 3 4 5 6 7 8"), None);
    }

    #[test]
    fn meminfo_and_keyed_values() {
        let meminfo = "MemTotal:       16384 kB\nMemAvailable:    8192 kB\nHugePages_Total:    4\n";
        assert_eq!(meminfo_bytes(meminfo, "MemTotal"), Some(16_384 * 1024));
        assert_eq!(meminfo_bytes(meminfo, "HugePages_Total"), Some(4));
        assert_eq!(meminfo_bytes(meminfo, "Mem"), None);
        let stat = "usage_usec 500\nthrottled_usec 20\nnr_periods 3\n";
        assert_eq!(keyed(stat, "usage_usec"), Some(500));
        assert_eq!(keyed(stat, "throttled"), None);
        assert_eq!(keyed("usage_usec x", "usage_usec"), None);
        assert!(keyed_line(stat, "nr_periods").is_some());
    }

    #[test]
    fn psi_totals() {
        let psi = "some avg10=0.00 avg60=0.00 avg300=0.00 total=1234\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=99\n";
        assert_eq!(psi_some_total_ns(psi), Some(1_234_000));
        assert_eq!(psi_some_total_ns("full total=5"), None);
        assert_eq!(psi_some_total_ns("some avg10=1 total=x"), None);
    }

    #[test]
    fn diskstats_lines() {
        let line = "   8       0 sda 100 0 2000 30 50 0 4000 70 3 90 100 0 0 0 0";
        let disk = diskstats_line(line).unwrap();
        assert_eq!(disk.name, "sda");
        assert_eq!(
            (disk.reads, disk.sectors_read, disk.read_ms),
            (100, 2000, 30)
        );
        assert_eq!(
            (
                disk.writes,
                disk.sectors_written,
                disk.write_ms,
                disk.in_flight
            ),
            (50, 4000, 70, 3)
        );
        assert_eq!(diskstats_line("8 0 sda 1 2 3"), None);
        assert_eq!(
            diskstats_line("8 0 sda 1 0 x 30 50 0 4000 70 3 90 100"),
            None
        );
    }

    #[test]
    fn io_stat_totals() {
        let io =
            "8:0 rbytes=100 wbytes=50 rios=1 wios=2 dbytes=0 dios=0\n253:0 rbytes=10 wbytes=5\n";
        assert_eq!(io_stat_bytes(io), Some(165));
        assert_eq!(io_stat_bytes(""), Some(0));
        assert_eq!(io_stat_bytes("8:0 rbytes=x"), None);
    }

    #[test]
    fn sizes_and_mount_paths() {
        assert_eq!(size_bytes("32K"), Some(32 << 10));
        assert_eq!(size_bytes("2M\n"), Some(2 << 20));
        assert_eq!(size_bytes("1G"), Some(1 << 30));
        assert_eq!(size_bytes("4096"), Some(4096));
        assert_eq!(size_bytes("K"), None);
        assert_eq!(
            unescape_mount_path("/mnt/with\\040space"),
            "/mnt/with space"
        );
        assert_eq!(unescape_mount_path("/trailing\\04"), "/trailing\\04");
    }
}
