use super::*;
use std::sync::atomic::{AtomicUsize, Ordering as StdOrdering};
use std::thread;

#[cfg(not(miri))]
const PAIRS: u32 = 10_000;
#[cfg(miri)]
const PAIRS: u32 = 40;

#[test]
fn send_before_take_and_take_before_send_both_deliver() {
    let (sender, mut receiver) = one_shot();
    sender.send(7);
    assert!(receiver.is_set());
    assert_eq!(receiver.try_take(), Ok(Some(7)));

    let (sender, mut receiver) = one_shot();
    assert!(!receiver.is_set());
    assert_eq!(receiver.try_take(), Ok(None));
    sender.send(8);
    assert_eq!(receiver.try_take(), Ok(Some(8)));
}

#[test]
fn a_dropped_sender_abandons() {
    let (sender, mut receiver) = one_shot::<u32>();
    assert_eq!(receiver.try_take(), Ok(None));
    drop(sender);
    assert!(!receiver.is_set());
    assert_eq!(receiver.try_take(), Err(RecvError::Abandoned));
}

#[test]
fn a_value_is_taken_once_and_stays_set() {
    let (sender, mut receiver) = one_shot();
    sender.send("done");
    assert_eq!(receiver.try_take(), Ok(Some("done")));
    assert!(receiver.is_set(), "set stays true after the take");
    assert_eq!(receiver.try_take(), Err(RecvError::Abandoned));
}

#[test]
fn ten_thousand_pairs_across_two_threads_each_deliver_once() {
    let (to_sender, senders) = std::sync::mpsc::channel();
    let (to_receiver, receivers) = std::sync::mpsc::channel();
    for id in 0..PAIRS {
        let (sender, receiver) = one_shot();
        to_sender.send((id, sender)).unwrap();
        to_receiver.send((id, receiver)).unwrap();
    }
    let sending = thread::spawn(move || {
        while let Ok((id, sender)) = senders.try_recv() {
            sender.send(id);
        }
    });
    let receiving = thread::spawn(move || {
        let mut delivered = 0;
        while let Ok((id, mut receiver)) = receivers.try_recv() {
            loop {
                match receiver.try_take() {
                    Ok(Some(value)) => {
                        assert_eq!(value, id);
                        delivered += 1;
                        break;
                    }
                    Ok(None) => thread::yield_now(),
                    Err(error) => panic!("abandoned: {error}"),
                }
            }
        }
        delivered
    });
    sending.join().unwrap();
    assert_eq!(receiving.join().unwrap(), PAIRS);
}

struct Counted(Arc<AtomicUsize>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, StdOrdering::SeqCst);
    }
}

#[test]
fn a_value_is_dropped_once_whoever_goes_first() {
    let drops = Arc::new(AtomicUsize::new(0));
    // The receiver goes first.
    let (sender, receiver) = one_shot();
    drop(receiver);
    sender.send(Counted(drops.clone()));
    assert_eq!(drops.load(StdOrdering::SeqCst), 1);
    // The receiver goes after the send, never having taken.
    let (sender, receiver) = one_shot();
    sender.send(Counted(drops.clone()));
    drop(receiver);
    assert_eq!(drops.load(StdOrdering::SeqCst), 2);
    // The value is taken and dropped by its owner.
    let (sender, mut receiver) = one_shot();
    sender.send(Counted(drops.clone()));
    drop(receiver.try_take().unwrap());
    assert_eq!(drops.load(StdOrdering::SeqCst), 3);
}

#[test]
fn senders_race_their_receivers_dropping() {
    for _ in 0..PAIRS / 10 {
        let drops = Arc::new(AtomicUsize::new(0));
        let (sender, receiver) = one_shot();
        let sending = {
            let drops = drops.clone();
            thread::spawn(move || sender.send(Counted(drops)))
        };
        let dropping = thread::spawn(move || drop(receiver));
        sending.join().unwrap();
        dropping.join().unwrap();
        assert_eq!(drops.load(StdOrdering::SeqCst), 1);
    }
}

#[cfg(loom)]
mod loom_tests {
    use super::super::*;
    use ::loom::thread;

    // A poll never sees a half-written value: a take that finds `SET` finds the whole value, and a send is seen by a
    // take that follows the join.
    #[test]
    fn loom_try_take_never_sees_a_value_before_it_is_complete() {
        ::loom::model(|| {
            let (sender, mut receiver) = one_shot();
            let sending = thread::spawn(move || sender.send((1_u32, 2_u32)));
            let mut seen = None;
            if let Ok(Some(pair)) = receiver.try_take() {
                assert_eq!(pair, (1, 2));
                seen = Some(pair);
            }
            sending.join().unwrap();
            match seen {
                Some(_) => assert_eq!(receiver.try_take(), Err(RecvError::Abandoned)),
                None => assert_eq!(receiver.try_take(), Ok(Some((1, 2)))),
            }
        });
    }

    #[test]
    fn loom_a_dropped_sender_racing_a_take_is_abandoned_or_empty() {
        ::loom::model(|| {
            let (sender, mut receiver) = one_shot::<u32>();
            let dropping = thread::spawn(move || drop(sender));
            let first = receiver.try_take();
            assert!(matches!(first, Ok(None) | Err(RecvError::Abandoned)));
            dropping.join().unwrap();
            assert_eq!(receiver.try_take(), Err(RecvError::Abandoned));
        });
    }

    #[test]
    fn loom_the_receiver_dropping_while_the_sender_sends_frees_the_value_once() {
        ::loom::model(|| {
            let (sender, receiver) = one_shot();
            let sending = thread::spawn(move || sender.send(String::from("value")));
            drop(receiver);
            sending.join().unwrap();
        });
    }
}
