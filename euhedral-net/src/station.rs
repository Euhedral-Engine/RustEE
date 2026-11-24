//! A source of the lattice that is driven by an inbox: the shape of one connection and of the listener.
//!
//! A [`Station`] is the part of a [`LatticeSource`] that every such source repeats. Anything may push an event
//! into its inbox (an MPSC queue) from any thread and returns at once: the frame an I/O continuation returns
//! (see [`Post`]), a `Responder`, the stop. The source's `request`, which only a worker calls, pushes the station's one *key frame* to the lattice
//! while the inbox holds something and the key is home. The key's `execute` drains the inbox and hands each event
//! to the owner's [`Drive::run`]; its `recycle` sends it home. So **at most one worker is ever inside an owner's
//! `run`**, the owner's state can be plain (a `SourceCell` that only `run` enters), and two owners are two sources
//! that run in parallel. This is how `AsyncIo` is built (`euhedral-io`, `ReapFrame` and its `key`), and the
//! rules in the `LatticeSource` documentation hold: `pull` hands out nothing, the key leaves only through
//! `request`, and nothing in the owner's visits is heavy.
//!
//! A station completes with its owner ([`Station::complete`]): the lattice then lets go of the source. Events
//! pushed after that are never run.
//!
//! The cost of the shape: an event waits for the next visit of a worker to its source (the lattice calls `request`
//! on every source at every visit, so the cost of a visit grows with the number of live sources, by one atomic read
//! for each that has nothing), and a source is attached to and detached from the lattice at run time (a lock of
//! the registry and a pass over its handles, once per connection).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::{
    DownstreamSlot, KeySlot, LatticeReceiver, LatticeSource, SourceCell,
};
use euhedral_data_structures::queues::{Consumer, MpscQueue, Multi, Producer, Single};

/// What a station's owner does with an event.
pub(crate) trait Drive: Send + Sync + 'static {
    type Event: Send + 'static;

    /// The owner's station.
    fn station(&self) -> &Station<Self>
    where
        Self: Sized;

    /// Runs one event. Called by the key frame only, so never twice at once.
    fn run(self: &Arc<Self>, event: Self::Event);
}

/// The inbox, the key and the downstream of one source.
pub(crate) struct Station<T: Drive> {
    producer: Producer<T::Event, Multi>,
    inbox: SourceCell<Consumer<T::Event, Single>>,
    key: KeySlot<Key<T>>,
    downstream: DownstreamSlot,
    /// Counts the pushes of the key, so that each is routed by a hash of its own.
    seed: AtomicU64,
}

impl<T: Drive> Station<T> {
    pub(crate) fn new(owner: Weak<T>) -> Self {
        let queue = MpscQueue::<T::Event>::with_pool(16, 2);
        Self {
            producer: queue.producer(),
            inbox: SourceCell::new(queue.consumer()),
            key: KeySlot::with_key(Key::new(owner)),
            downstream: DownstreamSlot::new(),
            seed: AtomicU64::new(0),
        }
    }

    /// Pushes `event` into the inbox; the key runs it at a later visit. Never waits.
    pub(crate) fn post(&self, event: T::Event) {
        // The queue is unbounded: it refuses nothing.
        let _ = self.producer.offer(event);
    }

    /// The source's `add_downstream`.
    pub(crate) fn add_downstream(
        &self,
        me: Arc<dyn LatticeSource>,
        downstream: Arc<dyn LatticeReceiver>,
    ) {
        self.downstream.attach(me, downstream);
    }

    /// The source's `request`: pushes the key if the inbox holds something and the key is home.
    pub(crate) fn request(&self, demand: u64) {
        if demand == 0 || self.producer.is_empty() {
            return;
        }
        let Some(downstream) = self.downstream.receiver() else {
            return;
        };
        let Some(mut key) = self.key.take() else {
            return;
        };
        key.arm(self.seed.fetch_add(1, Ordering::Relaxed));
        downstream.push(key);
    }

    /// Finishes the source: the lattice lets go of it. Atomics only, callable from the key itself.
    pub(crate) fn complete(&self) {
        self.downstream.complete();
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.downstream.is_complete()
    }

    /// The next event of the inbox, taken by the test that stands in for the key.
    #[cfg(test)]
    pub(crate) fn take_event(&self) -> Option<T::Event> {
        self.inbox.enter().poll()
    }

    /// Whether the inbox holds an event.
    #[cfg(test)]
    pub(crate) fn has_event(&self) -> bool {
        !self.producer.is_empty()
    }
}

/// The key frame of a station: its `execute` runs the owner's events.
struct Key<T: Drive> {
    header: FrameHeader,
    owner: Weak<T>,
}

impl<T: Drive> Key<T> {
    fn new(owner: Weak<T>) -> Box<Self> {
        Box::new(Self {
            header: FrameHeader::new(0x5747_4154_494F_4E00),
            owner,
        })
    }

    /// Unordered, and a new routing hash for each push: the key may run on any worker.
    fn arm(&mut self, seed: u64) {
        self.header.randomize_hash(seed.wrapping_add(1));
    }
}

impl<T: Drive> Frame for Key<T> {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        let Some(owner) = self.owner.upgrade() else {
            return Ok(());
        };
        // The key is one, and it is away while this runs: the entry cannot find the cell taken.
        let mut inbox = owner.station().inbox.enter();
        while let Some(event) = inbox.poll() {
            owner.run(event);
        }
        Ok(())
    }

    /// The key goes home. (An owner that is gone has no home: the frame is dropped.)
    fn recycle(self: Box<Self>) {
        if let Some(owner) = self.owner.upgrade() {
            owner.station().key.home(self);
        }
    }
}

impl<T: Drive> Drop for Key<T> {
    /// A key dropped without having run (a lattice that discarded it) must not strand the owner: a new one goes
    /// home. A key that is homed does not get here, and there is no other: the slot is empty.
    fn drop(&mut self) {
        if let Some(owner) = self.owner.upgrade() {
            owner
                .station()
                .key
                .home_replacement(Key::new(self.owner.clone()), |key| key.owner = Weak::new());
        }
    }
}

/// A frame that pushes an event into an owner's inbox when it runs: what a timer or an I/O continuation hands the
/// lattice for an event of a station. An I/O continuation posts nothing itself: the owner hears that the
/// operation ended only when this frame runs, which is after the `AsyncIo` handed it out and stopped counting the
/// operation as pending. A frame dropped without having run (a lattice that discarded it) posts its event as it
/// goes, so that the operation's end is never lost.
pub(crate) struct Post<T: Drive> {
    header: FrameHeader,
    owner: Arc<T>,
    event: Option<T::Event>,
}

impl<T: Drive> Post<T> {
    pub(crate) fn new(owner: Arc<T>, event: T::Event) -> Box<Self> {
        Box::new(Self {
            header: FrameHeader::new(0),
            owner,
            event: Some(event),
        })
    }
}

impl<T: Drive> Drop for Post<T> {
    fn drop(&mut self) {
        if let Some(event) = self.event.take() {
            self.owner.station().post(event);
        }
    }
}

impl<T: Drive> Frame for Post<T> {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(event) = self.event.take() {
            self.owner.station().post(event);
        }
        Ok(())
    }
}
