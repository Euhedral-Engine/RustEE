//! A stack that any thread pushes to and one thread takes whole, then seals (Java's `StageFrame.externals`: a CAS
//! stack closed by `getAndSet(sentinel)`).
//!
//! - **Intrusive:** the nodes are the caller's own, pooled objects, each holding its link, so a push allocates nothing.
//!   A node must stay where it is until the stack gives it back.
//! - **Sealed by a sentinel:** [`seal`](TakeAllStack::seal) swaps the head for a sentinel that names why the stack is
//!   closed, and hands back everything pushed so far. A push after that fails and returns the sentinel, so the pusher
//!   acts at once on what the sealing meant. Every node ends in exactly one place: in the list a seal returned, or
//!   refused by the push that tried it.
//! - **No ABA:** nothing is popped one at a time. The list is taken whole, by the one thread whose swap closed it.

#![allow(
    unsafe_code,
    reason = "the nodes are the caller's, linked by their own cells; a node is read only after the sealing swap handed its list to one thread"
)]

use std::ptr::{self, NonNull};

use crate::sync::{AtomicPtr, Ordering, Owned};

/// Why a stack was sealed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sealed {
    /// The producer of what the nodes wait for published it.
    Submitted,
    /// The producer can never publish it.
    Swept,
}

const SUBMITTED: usize = 1;
const SWEPT: usize = 2;

fn sentinel<N>(sealed: Sealed) -> *mut N {
    ptr::without_provenance_mut(match sealed {
        Sealed::Submitted => SUBMITTED,
        Sealed::Swept => SWEPT,
    })
}

fn sealed_by<N>(head: *mut N) -> Option<Sealed> {
    match head.addr() {
        SUBMITTED => Some(Sealed::Submitted),
        SWEPT => Some(Sealed::Swept),
        _ => None,
    }
}

/// The link a node of a [`TakeAllStack`] holds to the next node of the stack it is in.
///
/// Written by the push, before the exchange that publishes the node; read by the thread whose seal took the list.
pub(crate) struct Link<N>(Owned<*mut N>);

// SAFETY: the pointer is only read and written under the stack's ordering, and only dereferenced by the stack, whose
// `push` contract makes the node valid.
unsafe impl<N> Send for Link<N> {}
// SAFETY: as above.
unsafe impl<N> Sync for Link<N> {}

impl<N> Link<N> {
    pub(crate) fn new() -> Self {
        Self(Owned::new(ptr::null_mut()))
    }

    /// # Safety
    ///
    /// The node is not yet visible to any other thread.
    unsafe fn set(&self, next: *mut N) {
        // SAFETY: the caller's contract.
        unsafe { self.0.set(next) }
    }

    /// # Safety
    ///
    /// The calling thread took the list this node is in, and no thread writes the link.
    unsafe fn get(&self) -> *mut N {
        // SAFETY: the caller's contract.
        unsafe { self.0.get() }
    }
}

impl<N> Default for Link<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// A node of a [`TakeAllStack`].
///
/// # Safety
///
/// [`link`](Self::link) returns a link that belongs to this node alone and that nothing but the stack uses.
pub(crate) unsafe trait Linked: Sized {
    /// The node's link to the next node of the stack it is in.
    fn link(&self) -> &Link<Self>;
}

/// See the [module documentation](self).
pub(crate) struct TakeAllStack<N: Linked> {
    /// Null (open, empty), a node (open), or a sentinel (sealed).
    head: AtomicPtr<N>,
}

impl<N: Linked> TakeAllStack<N> {
    /// An open, empty stack.
    pub(crate) fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Opens the stack again, empty. Only while no thread uses it, and after the seal that closed it took its nodes.
    pub(crate) fn reopen(&self) {
        self.head.store(ptr::null_mut(), Ordering::Release);
    }

    /// Why the stack is sealed, if it is.
    pub(crate) fn sealed(&self) -> Option<Sealed> {
        sealed_by(self.head.load(Ordering::Acquire))
    }

    /// Pushes `node`, or returns why the stack is sealed (the node is then not in it).
    ///
    /// # Safety
    ///
    /// `node` is valid, and stays valid and in place until the stack hands it back (in a seal's list, or by this
    /// call's `Err`). It is in no stack.
    pub(crate) unsafe fn push(&self, node: NonNull<N>) -> Result<(), Sealed> {
        let mut head = self.head.load(Ordering::Acquire);
        loop {
            if let Some(sealed) = sealed_by(head) {
                return Err(sealed);
            }
            // SAFETY: `node` is valid (the caller's contract), and no other thread can see it before the exchange
            // below publishes it.
            unsafe { node.as_ref().link().set(head) };
            match self.head.compare_exchange_weak(
                head,
                node.as_ptr(),
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(now) => head = now,
            }
        }
    }

    /// Seals the stack and takes everything pushed so far, newest first. A stack that is sealed already stays as it
    /// was and yields nothing: the sealing that came first wins.
    pub(crate) fn seal(&self, sealed: Sealed) -> Taken<N> {
        let mut head = self.head.load(Ordering::Acquire);
        loop {
            if sealed_by(head).is_some() {
                return Taken {
                    next: ptr::null_mut(),
                };
            }
            match self.head.compare_exchange_weak(
                head,
                sentinel(sealed),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Taken { next: head },
                Err(now) => head = now,
            }
        }
    }
}

impl<N: Linked> std::fmt::Debug for TakeAllStack<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TakeAllStack")
            .field("sealed", &self.sealed())
            .finish_non_exhaustive()
    }
}

/// The nodes a [`seal`](TakeAllStack::seal) took, newest first. Each node is yielded after its link was read, so the
/// receiver of a node may let it go at once.
pub(crate) struct Taken<N: Linked> {
    next: *mut N,
}

impl<N: Linked> Iterator for Taken<N> {
    type Item = NonNull<N>;

    fn next(&mut self) -> Option<NonNull<N>> {
        let node = NonNull::new(self.next)?;
        // SAFETY: the seal's swap gave this thread the whole list, and `push`'s contract keeps every node valid until
        // it is yielded here. The nodes after this one are not yielded yet, so they are valid too.
        self.next = unsafe { node.as_ref().link().get() };
        Some(node)
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};

    struct Node {
        link: Link<Node>,
        id: usize,
    }

    // SAFETY: the link is the node's own, used by the stack alone.
    unsafe impl Linked for Node {
        fn link(&self) -> &Link<Node> {
            &self.link
        }
    }

    fn nodes(count: usize) -> Vec<Node> {
        (0..count)
            .map(|id| Node {
                link: Link::new(),
                id,
            })
            .collect()
    }

    fn at(node: &Node) -> NonNull<Node> {
        NonNull::from(node)
    }

    #[test]
    fn a_seal_takes_the_nodes_newest_first_and_later_pushes_are_refused() {
        let stack = TakeAllStack::new();
        let nodes = nodes(3);
        for node in &nodes {
            // SAFETY: the nodes outlive the stack's use and are pushed once.
            let pushed = unsafe { stack.push(at(node)) };
            assert_eq!(pushed, Ok(()));
        }
        assert_eq!(stack.sealed(), None);
        let taken: Vec<usize> = stack
            .seal(Sealed::Submitted)
            // SAFETY: the node is valid.
            .map(|node| unsafe { node.as_ref().id })
            .collect();
        assert_eq!(taken, [2, 1, 0]);
        assert_eq!(stack.sealed(), Some(Sealed::Submitted));
        let late = Box::new(Node {
            link: Link::new(),
            id: 9,
        });
        // SAFETY: the box is valid.
        let refused = unsafe { stack.push(at(&late)) };
        assert_eq!(
            refused,
            Err(Sealed::Submitted),
            "a push after the seal learns why"
        );
        assert_eq!(
            stack.seal(Sealed::Swept).count(),
            0,
            "the first seal stands"
        );
        assert_eq!(stack.sealed(), Some(Sealed::Submitted));
        stack.reopen();
        assert_eq!(stack.sealed(), None);
        // SAFETY: the box is valid and in no stack.
        let pushed = unsafe { stack.push(at(&late)) };
        assert_eq!(pushed, Ok(()));
        assert_eq!(stack.seal(Sealed::Swept).count(), 1);
    }

    /// Eight pushers against one seal: every node is taken or refused, exactly once.
    #[test]
    #[allow(clippy::disallowed_methods)] // The test's own pushers.
    fn pushes_racing_a_seal_are_either_taken_or_handed_back() {
        let rounds = if cfg!(miri) { 2 } else { 200 };
        let per_thread = if cfg!(miri) { 4 } else { 50 };
        for _ in 0..rounds {
            let stack = Arc::new(TakeAllStack::<Node>::new());
            let all = Arc::new(nodes(8 * per_thread));
            let refused = Arc::new(AtomicUsize::new(0));
            let pushers: Vec<_> = (0..8)
                .map(|t| {
                    let (stack, all, refused) = (stack.clone(), all.clone(), refused.clone());
                    std::thread::spawn(move || {
                        for i in 0..per_thread {
                            let node = &all[t * per_thread + i];
                            // SAFETY: the node lives in `all`, which outlives the threads, and is pushed once.
                            if unsafe { stack.push(at(node)) }.is_err() {
                                refused.fetch_add(1, StdOrdering::Relaxed);
                            }
                        }
                    })
                })
                .collect();
            let mut seen = vec![0u8; all.len()];
            let mut taken = 0;
            for node in stack.seal(Sealed::Submitted) {
                // SAFETY: the node is valid.
                seen[unsafe { node.as_ref().id }] += 1;
                taken += 1;
            }
            for pusher in pushers {
                pusher.join().unwrap();
            }
            assert_eq!(taken + refused.load(StdOrdering::Relaxed), all.len());
            assert!(seen.iter().all(|&n| n <= 1), "no node is taken twice");
        }
    }
}

#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;
    use loom::sync::Arc;
    use loom::sync::atomic::AtomicUsize;
    use loom::thread;

    struct Node {
        link: Link<Node>,
        id: usize,
    }

    // SAFETY: the link is the node's own, used by the stack alone.
    unsafe impl Linked for Node {
        fn link(&self) -> &Link<Node> {
            &self.link
        }
    }

    fn node(id: usize) -> Box<Node> {
        Box::new(Node {
            link: Link::new(),
            id,
        })
    }

    #[test]
    fn loom_two_pushes_and_a_seal_account_for_every_node_once() {
        loom::model(|| {
            let stack = Arc::new(TakeAllStack::<Node>::new());
            let nodes = Arc::new([node(0), node(1)]);
            let refused = Arc::new(AtomicUsize::new(0));
            let pushers: Vec<_> = (0..2)
                .map(|i| {
                    let (stack, nodes, refused) = (stack.clone(), nodes.clone(), refused.clone());
                    thread::spawn(move || {
                        // SAFETY: the node lives in `nodes`, which outlives the thread, and is pushed once.
                        if unsafe { stack.push(NonNull::from(&*nodes[i])) }.is_err() {
                            refused.fetch_add(1, loom::sync::atomic::Ordering::SeqCst);
                        }
                    })
                })
                .collect();
            let mut taken = 0;
            let mut ids = 0;
            for n in stack.seal(Sealed::Submitted) {
                taken += 1;
                // SAFETY: the node is valid.
                ids += 1 << unsafe { n.as_ref().id };
            }
            for pusher in pushers {
                pusher.join().unwrap();
            }
            assert_eq!(
                taken + refused.load(loom::sync::atomic::Ordering::SeqCst),
                2
            );
            assert!(ids <= 3);
        });
    }
}
