use super::{Frame, FrameHeader};

/// A frame that does nothing, for priming sequences such as filling queues.
#[derive(Debug)]
pub struct DummyFrame {
    header: FrameHeader,
}

impl DummyFrame {
    pub const INSTANCE: DummyFrame = DummyFrame {
        header: FrameHeader::new(0),
    };
}

impl Frame for DummyFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }
}
