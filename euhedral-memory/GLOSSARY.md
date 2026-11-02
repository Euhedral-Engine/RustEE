# euhedral-memory glossary

The implementation-level terms of [`euhedral-memory`](README.md): how the planner's plans reach real memory
and how memory moves between tiers. The terms a user plans with (locations, owners, claims, admission,
arenas, residents) and the lifecycles are in the [top-level glossary](../GLOSSARY.md). Paths are relative to
this crate. Entries run from the lowest layer up: backends, admission, tiers.

**backend** ([`Backend`](src/backend.rs), [`cuda`](src/cuda/mod.rs)): what allocates at a location. The
planner accounts and decides; a backend allocates and frees. The CUDA backends (feature `cuda`, the driver
loaded at run time) are `DeviceMemory`, the stream-ordered `StreamPool`, `Vmm` (exportable as a POSIX fd for
Vulkan) and `PinnedHost`, a capacity cache that keeps released buffers by power-of-two size class (64 KiB
and up) up to a byte limit and frees them only on `trim`.

**reaper** ([`Reaper`](src/backend.rs)): frees queued behind the device work that last used the memory (a
`Retired` test, such as a lane's marker). `push` never waits; `reap` frees what has retired, on the thread
that calls it, and a second thread reaping at the same moment returns early instead of waiting. A free is
ordinary work: run `reap` on the owner's lane, so its blocking time counts as that work's service time.

**tiered item** ([`PlanItem::tiered`](src/admission.rs)): an admission item of equal units over an ordered
list of locations, fastest first: its required units go to the first, the rest to the first location with
free room, then the next; units that fit nowhere are left out.

**tier** ([`TierPool`](src/tiers.rs)): a location with one planned allocation cut into equal slots. Free
slots are an elastic holding of the pool's owner; a retired slot's budget is back with the planner until
`restore` takes it again.
