use crate::frames::Frame;
use crate::internal::random;
use euhedral_hardware_utils::affinity;
use euhedral_hardware_utils::topology::CpuInfo;
use euhedral_hashing::hasher_api;
use std::fmt;

type Create<D, F> = Box<dyn FnMut(u64, D) -> F + Send>;
type Replace<D, F> = Box<dyn FnMut(D, &mut F) + Send>;

/// Makes frames from data, or refills a recycled frame with new data. Usually owned by a
/// [`FrameManager`](super::FrameManager).
///
/// Every frame a factory touches gets the factory's id hash and the CPU the factory was made on as
/// its origin. A frame that comes out unordered gets a fresh routing seed each time, so frames of one
/// factory spread out unless they are ordered.
pub struct FrameFactory<D, F> {
    id_hash: u64,
    seed: u64,
    origin: Option<CpuInfo>,
    create: Create<D, F>,
    replace: Replace<D, F>,
}

impl<D, F: Frame> FrameFactory<D, F> {
    /// `create` builds a frame from the factory's id hash and the data; `replace` puts new data into
    /// a recycled frame.
    pub fn new(
        create: impl FnMut(u64, D) -> F + Send + 'static,
        replace: impl FnMut(D, &mut F) + Send + 'static,
    ) -> Self {
        Self {
            id_hash: hasher_api::mix(random::next_u64()),
            seed: random::next_u64(),
            origin: affinity::current_cpu_info().copied(),
            create: Box::new(create),
            replace: Box::new(replace),
        }
    }

    /// The id hash given to every frame this factory creates.
    pub fn id_hash(&self) -> u64 {
        self.id_hash
    }

    /// The CPU this factory was made on, stamped on every frame as its origin.
    pub fn origin(&self) -> Option<CpuInfo> {
        self.origin
    }

    /// A new frame holding `data`.
    pub fn create(&mut self, data: D) -> Box<F> {
        let mut frame = Box::new((self.create)(self.id_hash, data));
        if !frame.header().is_ordered() {
            let seed = self.next_seed();
            frame.header_mut().randomize_hash(seed);
        }
        frame.header_mut().set_origin(self.origin);
        frame
    }

    /// Puts `data` into a recycled `frame`. A frame that was, or becomes, unordered gets a fresh
    /// routing seed.
    pub fn replace(&mut self, data: D, frame: &mut F) {
        let was_parallel = !frame.header().is_ordered();
        frame.header_mut().reset_hash();
        (self.replace)(data, frame);
        if was_parallel || !frame.header().is_ordered() {
            let seed = self.next_seed();
            frame.header_mut().randomize_hash(seed);
        }
        frame.header_mut().set_origin(self.origin);
    }

    fn next_seed(&mut self) -> u64 {
        let seed = self.seed;
        self.seed = seed.wrapping_add(1);
        seed
    }
}

impl<D, F> fmt::Debug for FrameFactory<D, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameFactory")
            .field("id_hash", &self.id_hash)
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::FrameHeader;

    struct Payload {
        header: FrameHeader,
        value: u32,
    }

    impl Frame for Payload {
        fn header(&self) -> &FrameHeader {
            &self.header
        }

        fn header_mut(&mut self) -> &mut FrameHeader {
            &mut self.header
        }
    }

    fn payload(id_hash: u64, value: u32, ordered: bool) -> Payload {
        let mut header = FrameHeader::new(id_hash);
        if !ordered {
            header.randomize_hash(17);
        }
        Payload { header, value }
    }

    #[test]
    fn create_stamps_the_factory_id_and_origin() {
        let mut factory = FrameFactory::new(
            |id, value| payload(id, value, true),
            |value, frame: &mut Payload| frame.value = value,
        );
        let frame = factory.create(5);
        assert_eq!(frame.value, 5);
        assert_eq!(frame.header().id_hash(), factory.id_hash());
        assert!(frame.header().is_ordered());
        assert_eq!(frame.header().origin(), factory.origin());
    }

    #[test]
    fn unordered_frames_get_a_fresh_seed_on_every_create() {
        let mut factory = FrameFactory::new(
            |id, value| payload(id, value, false),
            |value, frame: &mut Payload| frame.value = value,
        );
        let first = factory.create(1);
        let second = factory.create(2);
        assert!(!first.header().is_ordered());
        assert!(!second.header().is_ordered());
        assert_ne!(
            first.header().routing_hash(),
            second.header().routing_hash()
        );
    }

    #[test]
    fn replace_reseeds_a_frame_that_was_unordered() {
        let mut factory = FrameFactory::new(
            |id, value| payload(id, value, false),
            |value, frame: &mut Payload| frame.value = value,
        );
        let mut frame = factory.create(1);
        let before = frame.header().routing_hash();
        factory.replace(2, &mut frame);
        assert_eq!(frame.value, 2);
        assert!(!frame.header().is_ordered());
        assert_ne!(frame.header().routing_hash(), before);
    }

    #[test]
    fn replace_keeps_an_ordered_frame_ordered() {
        let mut factory = FrameFactory::new(
            |id, value| payload(id, value, true),
            |value, frame: &mut Payload| frame.value = value,
        );
        let mut frame = factory.create(1);
        frame.header_mut().set_origin(None);
        factory.replace(2, &mut frame);
        assert!(frame.header().is_ordered());
        assert_eq!(frame.header().origin(), factory.origin());
    }

    #[test]
    fn replace_reseeds_a_frame_the_callback_unorders() {
        let mut factory = FrameFactory::new(
            |id, value| payload(id, value, true),
            |value, frame: &mut Payload| {
                frame.value = value;
                frame.header.randomize_hash(1);
            },
        );
        let mut frame = factory.create(1);
        factory.replace(2, &mut frame);
        assert!(!frame.header().is_ordered());
        assert_ne!(
            frame.header().routing_hash(),
            frame.header().id_hash() ^ hasher_api::mix(1),
            "the factory's seed replaces the callback's"
        );
    }

    #[test]
    fn factories_get_distinct_id_hashes() {
        let create = |id, value| payload(id, value, true);
        let replace = |value, frame: &mut Payload| frame.value = value;
        let first = FrameFactory::new(create, replace);
        let second = FrameFactory::new(create, replace);
        assert_ne!(first.id_hash(), second.id_hash());
    }
}
