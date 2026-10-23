use crate::frames::{Frame, FrameError};
use crate::generics::AbstractExecutor;

/// The minimal [`AbstractExecutor`]: executes each frame and nothing more.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DefaultExecutor {
    cpu: Option<usize>,
}

impl DefaultExecutor {
    /// An executor not bound to any CPU.
    pub const fn new() -> Self {
        Self { cpu: None }
    }

    /// An executor for `cpu`.
    pub const fn on_cpu(cpu: usize) -> Self {
        Self { cpu: Some(cpu) }
    }
}

impl AbstractExecutor for DefaultExecutor {
    fn cpu(&self) -> Option<usize> {
        self.cpu
    }

    fn execute(&self, frame: &mut dyn Frame) -> Result<(), FrameError> {
        frame.execute()
    }

    fn hook_on_clone(&self, cpu: Option<usize>) -> Self {
        Self { cpu }
    }
}

#[cfg(test)]
mod tests;
