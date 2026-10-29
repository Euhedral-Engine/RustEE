//! Lock-free cells shared across threads.
//!
//! - [`SetOnce`]: a value installed at most once by a compare-and-set, read with an acquire load.

mod loom;
mod set_once;

pub use set_once::SetOnce;
