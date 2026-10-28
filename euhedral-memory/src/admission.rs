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
//! A **tiered** consumer ([`PlanItem::tiered`]) is a set of equal units (slots of a buffer, pages of a cache) that may
//! live in any of an ordered list of locations, fastest first (a device, then pinned host memory, then a window onto
//! a file): its first `min_first` units are required in the first location and placed as above; once everything fits,
//! its other units go to the first location that has free room for them, then the next, highest priority first. The
//! units that fit nowhere are left out ([`Admission::tiers`] says where each went): the consumer runs with fewer.
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
    /// A tiered consumer's units beyond those in `needs` (see the module docs).
    pub tiers: Option<Tiers>,
}

/// The units of a tiered consumer placed after its first location's required ones.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tiers {
    /// Bytes of one unit.
    pub unit: u64,
    /// Units beyond the required ones.
    pub extra: u32,
    /// Where they may go, fastest first (the first is where the required units are).
    pub order: Vec<LocationId>,
}

impl PlanItem {
    pub fn required(consumer: impl Into<String>, location: LocationId, bytes: u64) -> Self {
        PlanItem {
            consumer: consumer.into(),
            priority: Priority(u16::MAX),
            required: true,
            needs: vec![(location, bytes)],
            spill: None,
            tiers: None,
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
            tiers: None,
        }
    }

    /// A tiered consumer: `units` units of `unit` bytes over `order` (fastest first), the first `min_first` of them
    /// required in `order[0]`.
    pub fn tiered(
        consumer: impl Into<String>,
        priority: Priority,
        required: bool,
        unit: u64,
        units: u32,
        min_first: u32,
        order: Vec<LocationId>,
    ) -> Self {
        let first = order.first().copied().unwrap_or(LocationId(0));
        let min_first = min_first.min(units);
        PlanItem {
            consumer: consumer.into(),
            priority,
            required,
            needs: vec![(first, unit * u64::from(min_first))],
            spill: None,
            tiers: Some(Tiers {
                unit,
                extra: units - min_first,
                order,
            }),
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
    /// Per item, a tiered consumer's units per location (its required ones included), in its order; empty for the
    /// others and for a dropped one.
    pub tiers: Vec<Vec<(LocationId, u32)>>,
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
        let mut bytes = total(&placed);
        let tiers = self.place_tiers(request, &placed, &room, &mut bytes);
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
                tiers,
                orders,
                reservation: None,
            });
        }
        let claims = self.reserve_many(request.owner, &bytes)?;
        Ok(Admission {
            placed,
            bytes,
            tiers,
            orders,
            reservation: Some(Reservation { claims }),
        })
    }

    /// The tiered consumers' units beyond their required ones, highest priority first, each to the first location of
    /// its order with free room (not reclaimable room: these never order a give-back); added to `bytes`.
    fn place_tiers(
        &self,
        request: &PlanRequest,
        placed: &[Placed],
        room: &[(u64, u64)],
        bytes: &mut [u64],
    ) -> Vec<Vec<(LocationId, u32)>> {
        let mut out = vec![Vec::new(); request.items.len()];
        let mut order: Vec<usize> = (0..request.items.len()).collect();
        // Highest priority first; ties: the item listed first.
        order.sort_by(|&a, &b| {
            request.items[b]
                .priority
                .cmp(&request.items[a].priority)
                .then(a.cmp(&b))
        });
        for i in order {
            let item = &request.items[i];
            let Some(t) = item.tiers.as_ref().filter(|_| placed[i] != Placed::Dropped) else {
                continue;
            };
            let unit = t.unit.max(1);
            let mut units: Vec<(LocationId, u32)> = t.order.iter().map(|&l| (l, 0)).collect();
            if let Some(first) = units.first_mut() {
                first.1 = u32::try_from(
                    item.needs
                        .iter()
                        .filter(|(l, _)| *l == first.0)
                        .map(|(_, b)| b / unit)
                        .sum::<u64>(),
                )
                .unwrap_or(u32::MAX);
            }
            let mut left = t.extra;
            for (loc, n) in &mut units {
                if left == 0 {
                    break;
                }
                let k = usize::from(loc.0);
                let (Some(&(free, _)), Some(used)) = (room.get(k), bytes.get_mut(k)) else {
                    continue;
                };
                let fit = u32::try_from(free.saturating_sub(*used) / unit).unwrap_or(u32::MAX);
                let take = fit.min(left);
                *n += take;
                *used += u64::from(take) * unit;
                left -= take;
            }
            out[i] = units;
        }
        out
    }
}
