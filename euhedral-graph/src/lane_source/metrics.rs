use std::sync::atomic::AtomicU64;

use euhedral_core::metrics::Histogram;

/// What a lane source records. Values are nanoseconds, except the depth.
#[derive(Debug)]
pub struct LaneMetrics {
    /// The time inside each `request` and each `pull`.
    pub service_ns: Histogram,
    /// The time inside `run`, per item kind (`W::KINDS`).
    pub item_ns: Box<[Histogram]>,
    /// From delivery to the item's first run.
    pub intake_to_run_ns: Histogram,
    /// The ready queue's depth at the end of each `request`.
    pub ready_depth: Histogram,
    /// Items skipped because they were not alive ([`LaneWork::is_alive`](super::LaneWork::is_alive)), per item kind
    /// (`W::KINDS`). A running count: reading it never scans a queue.
    pub skipped: Box<[AtomicU64]>,
    /// Items dropped without their `retired` call because an item took their retirement from the
    /// submitter (a misuse; the count says it happened).
    pub stranded: AtomicU64,
}

impl LaneMetrics {
    pub(crate) fn new(kinds: usize) -> Self {
        Self {
            service_ns: Histogram::new(),
            item_ns: (0..kinds).map(|_| Histogram::new()).collect(),
            intake_to_run_ns: Histogram::new(),
            ready_depth: Histogram::new(),
            skipped: (0..kinds).map(|_| AtomicU64::new(0)).collect(),
            stranded: AtomicU64::new(0),
        }
    }
}
