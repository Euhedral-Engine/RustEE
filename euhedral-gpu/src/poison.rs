//! A device's first fault, set once.

use std::sync::OnceLock;

use crate::GpuError;

/// The first fault of a device. Once set, later submissions fail fast with it.
#[derive(Debug, Default)]
pub struct Poison(OnceLock<GpuError>);

impl Poison {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `cause` unless a cause is already recorded: the first fault wins.
    pub fn set(&self, cause: GpuError) {
        let _ = self.0.set(cause);
    }

    pub fn get(&self) -> Option<GpuError> {
        self.0.get().cloned()
    }

    /// `Err(Poisoned(cause))` once poisoned.
    pub fn check(&self) -> Result<(), GpuError> {
        match self.0.get() {
            Some(cause) => Err(GpuError::Poisoned(Box::new(cause.clone()))),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuError;

    fn fault(code: i32) -> GpuError {
        GpuError::Fault {
            call: "test",
            code,
            name: "X".into(),
        }
    }

    #[test]
    fn poison_keeps_the_first_cause() {
        let p = Poison::new();
        assert_eq!(p.get(), None);
        assert_eq!(p.check(), Ok(()));
        p.set(fault(1));
        p.set(fault(2));
        assert_eq!(p.get(), Some(fault(1)));
        assert_eq!(p.check(), Err(GpuError::Poisoned(Box::new(fault(1)))));
    }
}
