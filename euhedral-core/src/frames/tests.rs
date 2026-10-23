use super::*;
use euhedral_hashing::hasher_api::mix;

struct TestFrame {
    header: FrameHeader,
    executed: bool,
}

impl TestFrame {
    fn new(id_hash: u64) -> Self {
        Self::with_kill_switch(id_hash, None)
    }

    fn with_kill_switch(id_hash: u64, kill_switch: Option<Arc<AtomicBool>>) -> Self {
        Self {
            header: FrameHeader::with_kill_switch(id_hash, kill_switch),
            executed: false,
        }
    }
}

impl Frame for TestFrame {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.executed = true;
        Ok(())
    }
}

struct DefaultFrame(FrameHeader);

impl Frame for DefaultFrame {
    fn header(&self) -> &FrameHeader {
        &self.0
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.0
    }
}

#[test]
fn a_new_header_is_ordered_alive_and_unplaced() {
    let frame = TestFrame::new(0x1234_5678);
    let header = frame.header();
    assert_eq!(header.id_hash(), 0x1234_5678);
    assert_eq!(header.routing_hash(), 0x1234_5678);
    assert!(header.is_ordered());
    assert!(frame.is_alive());
    assert_eq!(header.routing_policy(), RoutingPolicy::Anywhere);
    assert_eq!(header.origin(), None);
    assert!(header.kill_switch().is_none());
}

#[test]
fn a_header_with_a_kill_switch_dies_with_it() {
    let kill_switch = Arc::new(AtomicBool::new(false));
    let frame = TestFrame::with_kill_switch(0xab_cdef, Some(kill_switch.clone()));
    assert!(frame.header().is_ordered());
    assert!(frame.is_alive());
    assert!(Arc::ptr_eq(
        frame.header().kill_switch().unwrap(),
        &kill_switch
    ));

    kill_switch.store(true, Ordering::Release);
    assert!(!frame.is_alive());
}

#[test]
fn randomize_hash_mixes_the_seed_into_the_id_hash() {
    let mut frame = TestFrame::new(0x1000);
    frame.header_mut().randomize_hash(0x42);
    assert_eq!(frame.header().routing_hash(), 0x1000 ^ mix(0x42));
    assert!(!frame.header().is_ordered());

    let mut again = TestFrame::new(0x1000);
    again.header_mut().randomize_hash(0x42);
    assert_eq!(again.header().routing_hash(), frame.header().routing_hash());
}

#[test]
fn randomize_hash_with_seed_zero_still_unorders_the_frame() {
    assert_eq!(mix(0), 0, "the plain mix maps zero to itself");
    let mut frame = TestFrame::new(0x1000);
    frame.header_mut().randomize_hash(0);
    assert!(!frame.header().is_ordered());
}

#[test]
fn reset_hash_restores_the_id_hash() {
    let mut frame = TestFrame::new(0x2000);
    frame.header_mut().randomize_hash(99);
    assert!(!frame.header().is_ordered());

    frame.header_mut().reset_hash();
    assert_eq!(frame.header().routing_hash(), 0x2000);
    assert!(frame.header().is_ordered());
}

#[test]
fn routing_policy_is_configurable() {
    let mut frame = TestFrame::new(0x4000);
    for policy in [
        RoutingPolicy::SocketLocal,
        RoutingPolicy::CacheLocal,
        RoutingPolicy::Anywhere,
    ] {
        frame.header_mut().set_routing_policy(policy);
        assert_eq!(frame.header().routing_policy(), policy);
    }
}

#[test]
fn origin_is_settable_and_clearable() {
    let mut frame = TestFrame::new(0x5000);
    let cpu = CpuInfo {
        cpu: 1,
        core: 0,
        socket: 0,
    };
    frame.header_mut().set_origin(Some(cpu));
    assert_eq!(frame.header().origin(), Some(cpu));
    frame.header_mut().set_origin(None);
    assert_eq!(frame.header().origin(), None);
}

#[test]
fn kill_without_a_kill_switch_does_nothing() {
    let frame = TestFrame::new(0x6000);
    frame.kill();
    assert!(frame.is_alive());
}

#[test]
fn kill_sets_the_shared_switch() {
    let kill_switch = Arc::new(AtomicBool::new(false));
    let frame = TestFrame::with_kill_switch(0x7000, Some(kill_switch.clone()));
    let sibling = TestFrame::with_kill_switch(0x7001, Some(kill_switch.clone()));

    frame.kill();
    assert!(kill_switch.load(Ordering::Acquire));
    assert!(!frame.is_alive());
    assert!(!sibling.is_alive());
}

#[test]
fn execute_runs_the_override_or_the_no_op_default() {
    let mut frame = TestFrame::new(0x8000);
    assert!(!frame.executed);
    frame.execute().unwrap();
    assert!(frame.executed);

    let mut default = DefaultFrame(FrameHeader::new(0x8001));
    assert!(default.execute().is_ok());
}

#[test]
fn finalizers_without_a_pool_drop_the_frame() {
    Box::new(TestFrame::new(0x9000)).do_finally();
    Box::new(TestFrame::new(0x9001)).do_finally_with_error(FrameError::Cancelled);
    Box::new(TestFrame::new(0x9002)).recycle();
}

#[test]
fn frames_are_usable_as_trait_objects() {
    let mut frames: Vec<Box<dyn Frame>> = vec![
        Box::new(TestFrame::new(1)),
        Box::new(DefaultFrame(FrameHeader::new(2))),
        Box::new(DummyFrame::INSTANCE),
    ];
    for frame in &mut frames {
        frame.execute().unwrap();
    }
    let ids: Vec<u64> = frames.iter().map(|f| f.header().id_hash()).collect();
    assert_eq!(ids, [1, 2, 0]);
    for frame in frames {
        frame.do_finally();
    }
}

#[test]
fn a_frame_downcasts_back_to_its_type() {
    let frame: Box<dyn Frame> = Box::new(TestFrame::new(0xa000));
    let any: Box<dyn Any> = frame;
    let frame = any.downcast::<TestFrame>().ok().unwrap();
    assert_eq!(frame.header().id_hash(), 0xa000);
}

#[test]
fn the_dummy_frame_is_inert() {
    let mut dummy = DummyFrame::INSTANCE;
    assert_eq!(dummy.header().id_hash(), 0);
    assert!(dummy.header().is_ordered());
    assert!(dummy.is_alive());
    assert!(dummy.execute().is_ok());
}

#[test]
fn frame_errors_describe_themselves() {
    let cancelled = FrameError::Cancelled;
    assert!(cancelled.is_cancelled());
    assert_eq!(cancelled.to_string(), "frame cancelled");
    assert!(cancelled.source().is_none());

    let failed = FrameError::failed("stage blew up");
    assert!(!failed.is_cancelled());
    assert_eq!(failed.to_string(), "frame failed: stage blew up");
    assert_eq!(failed.source().unwrap().to_string(), "stage blew up");
}
