//! Admission: a plan's whole working set is placed before anything is allocated.
//!
//! A [`PlanRequest`] lists consumers, each with a priority, whether it is required, its bytes at its preferred
//! locations and optionally a spill placement (the same consumer with part of its memory on another location, for
//! example a buffer of slots on pinned host memory instead of the device). [`Planner::admit`] places them:
//!
//! 1. everything at its preferred placement, if it fits in what is free plus what lower-priority elastic owners
//!    could give back;
//! 2. else the spillable consumers move to their spill placement, lowest priority first, until it fits (spill before
//!    cutting a consumer);
//! 3. else optional consumers are dropped, lowest priority first;
//! 4. else the plan is refused, with the shortfall.
//!
//! An admitted plan comes with a [`Reservation`]: firm claims for its whole working set, from which its allocations
//! take their claims ([`Reservation::take`]). When give-back is needed first, the admission carries the orders and no
//! reservation: admit again once they are done (nothing waits).

use std::sync::Arc;

use crate::planner::{Claim, GiveBack, LocationId, OwnerId, Planner, Priority, Shortfall};

/// One consumer of a plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanItem {
    pub consumer: String,
    pub priority: Priority,
    /// A required item is never dropped (the plan is refused instead).
    pub required: bool,
    /// Bytes per location at the preferred placement.
    pub needs: Vec<(LocationId, u64)>,
    /// Bytes per location if spilled, if the consumer can spill.
    pub spill: Option<Vec<(LocationId, u64)>>,
}

impl PlanItem {
    pub fn required(consumer: impl Into<String>, location: LocationId, bytes: u64) -> Self {
        PlanItem {
            consumer: consumer.into(),
            priority: Priority(u16::MAX),
            required: true,
            needs: vec![(location, bytes)],
            spill: None,
        }
    }

    pub fn optional(
        consumer: impl Into<String>,
        priority: Priority,
        location: LocationId,
        bytes: u64,
    ) -> Self {
        PlanItem {
            consumer: consumer.into(),
            priority,
            required: false,
            needs: vec![(location, bytes)],
            spill: None,
        }
    }

    pub fn spills_to(mut self, spill: Vec<(LocationId, u64)>) -> Self {
        self.spill = Some(spill);
        self
    }
}

/// A plan to admit, for `owner` (the plan's owner: its reservation is accounted to it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanRequest {
    pub owner: OwnerId,
    pub items: Vec<PlanItem>,
}

/// Where an admitted item went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placed {
    Preferred,
    Spilled,
    Dropped,
}

/// The planner's answer.
#[derive(Debug)]
pub struct Admission {
    /// Per item of the request, in its order.
    pub placed: Vec<Placed>,
    /// Bytes per location of the admitted placement.
    pub bytes: Vec<u64>,
    /// Give-back orders issued; when not empty there is no reservation yet: admit again once they are done.
    pub orders: Vec<GiveBack>,
    pub reservation: Option<Reservation>,
}

impl Admission {
    pub fn dropped(&self) -> impl Iterator<Item = usize> + '_ {
        self.placed
            .iter()
            .enumerate()
            .filter(|(_, p)| **p == Placed::Dropped)
            .map(|(i, _)| i)
    }
}

/// Firm claims held for an admitted plan, per location. Allocations take their claims from it; what is left is given
/// back when the reservation drops (or [`Reservation::trim`]).
#[derive(Debug)]
pub struct Reservation {
    claims: Vec<Option<Claim>>,
}

impl Reservation {
    /// Takes `bytes` at `location` for `owner` (an allocation of the plan). `None` when the plan did not reserve that
    /// much there: the allocation was not planned, which is a bug to report, not a reason to allocate anyway.
    pub fn take(
        &mut self,
        owner: OwnerId,
        location: LocationId,
        bytes: u64,
        tag: &'static str,
    ) -> Option<Claim> {
        self.claims
            .get_mut(usize::from(location.0))?
            .as_mut()?
            .take(owner, bytes, tag)
    }

    /// Bytes still reserved at `location`.
    pub fn left(&self, location: LocationId) -> u64 {
        self.claims
            .get(usize::from(location.0))
            .and_then(Option::as_ref)
            .map_or(0, Claim::bytes)
    }

    /// Gives every unused byte back now.
    pub fn trim(&mut self) {
        for c in self.claims.iter_mut().flatten() {
            let b = c.bytes();
            c.shrink(b);
        }
    }
}

impl Planner {
    /// Places a plan (see the module docs).
    pub fn admit(self: &Arc<Self>, request: &PlanRequest) -> Result<Admission, Shortfall> {
        let n = self.locations();
        let priority = self.priority_of(request.owner);
        let room = self.room(priority);
        let mut placed = vec![Placed::Preferred; request.items.len()];
        let total = |placed: &[Placed]| -> Vec<u64> {
            let mut t = vec![0u64; n];
            for (item, p) in request.items.iter().zip(placed) {
                let list = match p {
                    Placed::Preferred => &item.needs,
                    Placed::Spilled => item.spill.as_ref().unwrap_or(&item.needs),
                    Placed::Dropped => continue,
                };
                for &(loc, b) in list {
                    t[usize::from(loc.0)] += b;
                }
            }
            t
        };
        let fits = |t: &[u64]| {
            t.iter()
                .zip(&room)
                .all(|(&need, &(free, reclaim))| need <= free + reclaim)
        };
        // Lowest priority first; ties: the item listed later goes first.
        let mut order: Vec<usize> = (0..request.items.len()).collect();
        order.sort_by(|&a, &b| {
            request.items[a]
                .priority
                .cmp(&request.items[b].priority)
                .then(b.cmp(&a))
        });
        if !fits(&total(&placed)) {
            for &i in &order {
                if request.items[i].spill.is_some() {
                    let mut trial = placed.clone();
                    trial[i] = Placed::Spilled;
                    // Spill only if it does not overfill the spill location.
                    let t = total(&trial);
                    let worse = t.iter().zip(&room).zip(total(&placed)).any(
                        |((&need, &(free, reclaim)), before)| {
                            need > before && need > free + reclaim
                        },
                    );
                    if !worse {
                        placed = trial;
                    }
                    if fits(&total(&placed)) {
                        break;
                    }
                }
            }
        }
        if !fits(&total(&placed)) {
            for &i in &order {
                if !request.items[i].required {
                    placed[i] = Placed::Dropped;
                    if fits(&total(&placed)) {
                        break;
                    }
                }
            }
        }
        let bytes = total(&placed);
        if let Some((loc, (&need, &(free, reclaim)))) = bytes
            .iter()
            .zip(&room)
            .enumerate()
            .find(|(_, (need, (free, reclaim)))| **need > free + reclaim)
        {
            return Err(Shortfall {
                location: LocationId(loc as u16),
                asked: need,
                free,
                reclaimable: reclaim,
            });
        }
        // Give-back first where the free memory alone does not cover it.
        let mut orders = Vec::new();
        for (loc, (&need, &(free, _))) in bytes.iter().zip(&room).enumerate() {
            if need > free {
                match self.claim_or_reclaim(
                    request.owner,
                    LocationId(loc as u16),
                    need,
                    "admission",
                ) {
                    crate::planner::ClaimResult::Reclaiming(o) => orders.extend(o),
                    crate::planner::ClaimResult::Granted(_) => {}
                    crate::planner::ClaimResult::Refused(s) => return Err(s),
                }
            }
        }
        if !orders.is_empty() {
            return Ok(Admission {
                placed,
                bytes,
                orders,
                reservation: None,
            });
        }
        let claims = self.reserve_many(request.owner, &bytes)?;
        Ok(Admission {
            placed,
            bytes,
            orders,
            reservation: Some(Reservation { claims }),
        })
    }
}
