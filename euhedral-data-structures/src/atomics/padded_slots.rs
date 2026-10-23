/// Storage shared by the padded array types: logical element `i` lives at physical slot
/// `(i + 1) * padding + i`, so every element has `padding` unused slots on each side.
#[derive(Debug)]
pub(crate) struct PaddedSlots<A> {
    pub(crate) padding: usize,
    pub(crate) array: Box<[A]>,
    pub(crate) bounds_check: bool,
    pow2: bool,
    length: usize,
}

impl<A: Default> PaddedSlots<A> {
    /// Slots of padding that keep an 8-byte element on its own 64-byte line.
    const MAX_PADDING: usize = 7;

    pub(crate) fn new(length: usize, bounds_check: bool, pad128: bool) -> Self {
        let mut padding = if pad128 {
            Self::MAX_PADDING * 2 + 2
        } else {
            Self::MAX_PADDING + 1
        };

        // Shrink the padding until the backing array fits in an `int`, as upstream does.
        let padded = loop {
            padding -= 1;
            assert!(padding > 0, "length {length} is too large to pad");
            let padded = (length as u64 + 1) * padding as u64 + length as u64;
            if padded <= i32::MAX as u64 {
                break padded as usize;
            }
        };

        Self {
            padding,
            array: (0..padded).map(|_| A::default()).collect(),
            bounds_check,
            pow2: length.is_power_of_two(),
            length,
        }
    }
}

impl<A> PaddedSlots<A> {
    pub(crate) fn length(&self) -> usize {
        self.length
    }

    pub(crate) fn slot(&self, idx: usize) -> &A {
        &self.array[self.physical_idx(idx)]
    }

    /// Physical slot of logical element `idx`, ignoring `bounds_check`.
    pub(crate) fn logical_to_physical(&self, idx: usize) -> usize {
        (idx + 1) * self.padding + idx
    }

    pub(crate) fn logical_slots(&self) -> impl Iterator<Item = &A> {
        (0..self.length).map(|idx| &self.array[self.logical_to_physical(idx)])
    }

    /// Maps an arbitrary hash or counter onto a physical slot. Power-of-two lengths mask the low bits;
    /// other lengths take `floorMod((int) raw, length)`, truncating to 32 bits first like upstream.
    pub(crate) fn physical_idx_for_raw(&self, raw_idx: i64) -> usize {
        assert!(self.length > 0, "cannot map an index onto an empty array");
        let logical = if self.pow2 {
            (raw_idx & (self.length as i64 - 1)) as usize
        } else {
            (raw_idx as i32).rem_euclid(self.length as i32) as usize
        };
        self.logical_to_physical(logical)
    }

    fn physical_idx(&self, idx: usize) -> usize {
        if !self.bounds_check {
            return idx;
        }
        assert!(
            idx < self.length,
            "Index {idx} out of bounds for length {}",
            self.length
        );
        self.logical_to_physical(idx)
    }
}
