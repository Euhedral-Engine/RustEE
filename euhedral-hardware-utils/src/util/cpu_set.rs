use std::fmt;

/// An immutable set of logical CPU ids (upstream's `UnmodifiableBitSet`).
///
/// The type has no mutators: every set operation returns a new set. The words are kept without
/// trailing zeros, so two equal sets always compare and hash equal no matter how they were built.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct CpuSet {
    words: Vec<u64>,
}

impl CpuSet {
    /// The empty set.
    pub const fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// Builds a set from little-endian words: word `i`, bit `b` is CPU `64 * i + b`.
    pub fn from_words(words: &[u64]) -> Self {
        Self::from_owned_words(words.to_vec())
    }

    pub(crate) fn from_owned_words(mut words: Vec<u64>) -> Self {
        while words.last() == Some(&0) {
            words.pop();
        }
        Self { words }
    }

    /// The set `{start, start + 1, ..., end - 1}`.
    pub fn range(start: usize, end: usize) -> Self {
        let mut words = Vec::new();
        for cpu in start..end {
            set_bit(&mut words, cpu);
        }
        Self::from_owned_words(words)
    }

    /// Little-endian words without trailing zero words (empty for the empty set).
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn contains(&self, cpu: usize) -> bool {
        self.words
            .get(cpu >> 6)
            .is_some_and(|word| word & (1 << (cpu & 63)) != 0)
    }

    /// Number of CPUs in the set (`BitSet.cardinality()`).
    pub fn len(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// One past the highest CPU in the set, or 0 when empty (`BitSet.length()`).
    pub fn span(&self) -> usize {
        match self.words.last() {
            Some(last) => (self.words.len() - 1) * 64 + (64 - last.leading_zeros() as usize),
            None => 0,
        }
    }

    pub fn first(&self) -> Option<usize> {
        self.iter().next()
    }

    pub fn last(&self) -> Option<usize> {
        self.span().checked_sub(1)
    }

    /// CPUs in ascending order.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            words: &self.words,
            index: 0,
            current: self.words.first().copied().unwrap_or(0),
        }
    }

    pub fn union(&self, other: &CpuSet) -> CpuSet {
        let (long, short) = if self.words.len() >= other.words.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut words = long.words.clone();
        for (word, extra) in words.iter_mut().zip(&short.words) {
            *word |= extra;
        }
        CpuSet { words }
    }

    pub fn intersection(&self, other: &CpuSet) -> CpuSet {
        let words = self
            .words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| a & b)
            .collect();
        CpuSet::from_owned_words(words)
    }

    /// CPUs in `self` that are not in `other`.
    pub fn difference(&self, other: &CpuSet) -> CpuSet {
        let words = self
            .words
            .iter()
            .enumerate()
            .map(|(i, word)| word & !other.words.get(i).copied().unwrap_or(0))
            .collect();
        CpuSet::from_owned_words(words)
    }

    pub fn intersects(&self, other: &CpuSet) -> bool {
        self.words.iter().zip(&other.words).any(|(a, b)| a & b != 0)
    }

    /// The number of CPUs in both sets, counted word by word without building the intersection.
    pub fn intersection_len(&self, other: &CpuSet) -> usize {
        self.words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| (a & b).count_ones() as usize)
            .sum()
    }

    /// The `n`th CPU (counting from 0, in ascending order) that is in both sets, found word by word without building
    /// the intersection; `None` if there are not that many.
    pub fn nth_of_intersection(&self, other: &CpuSet, mut n: usize) -> Option<usize> {
        for (index, (a, b)) in self.words.iter().zip(&other.words).enumerate() {
            let mut word = a & b;
            let count = word.count_ones() as usize;
            if n >= count {
                n -= count;
                continue;
            }
            // Clear the lowest `n` set bits; the lowest remaining one is the answer.
            for _ in 0..n {
                word &= word - 1;
            }
            return Some(index * 64 + word.trailing_zeros() as usize);
        }
        None
    }

    pub fn is_subset(&self, other: &CpuSet) -> bool {
        self.difference(other).is_empty()
    }

    /// A copy with `cpu` added.
    pub fn with(&self, cpu: usize) -> CpuSet {
        let mut words = self.words.clone();
        set_bit(&mut words, cpu);
        CpuSet { words }
    }

    /// A copy with `cpu` removed.
    pub fn without(&self, cpu: usize) -> CpuSet {
        let mut words = self.words.clone();
        if let Some(word) = words.get_mut(cpu >> 6) {
            *word &= !(1 << (cpu & 63));
        }
        CpuSet::from_owned_words(words)
    }
}

pub(crate) fn set_bit(words: &mut Vec<u64>, cpu: usize) {
    let index = cpu >> 6;
    if words.len() <= index {
        words.resize(index + 1, 0);
    }
    words[index] |= 1 << (cpu & 63);
}

impl FromIterator<usize> for CpuSet {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut words = Vec::new();
        for cpu in iter {
            set_bit(&mut words, cpu);
        }
        CpuSet::from_owned_words(words)
    }
}

impl<'a> IntoIterator for &'a CpuSet {
    type Item = usize;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

/// Ascending iterator over a [`CpuSet`].
#[derive(Clone, Debug)]
pub struct Iter<'a> {
    words: &'a [u64],
    index: usize,
    current: u64,
}

impl Iterator for Iter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.current != 0 {
                let bit = self.current.trailing_zeros() as usize;
                self.current &= self.current - 1;
                return Some(self.index * 64 + bit);
            }
            self.index += 1;
            self.current = *self.words.get(self.index)?;
        }
    }
}

/// `{0, 2, 5}`, the format of Java's `BitSet.toString()`.
impl fmt::Display for CpuSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        for (i, cpu) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{cpu}")?;
        }
        f.write_str("}")
    }
}

impl fmt::Debug for CpuSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CpuSet{self}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_sets_built_differently_compare_equal() {
        let a = CpuSet::from_words(&[0b101, 0, 0]);
        let b: CpuSet = [2, 0].into_iter().collect();
        assert_eq!(a, b);
        assert_eq!(a.words(), &[0b101]);
        assert_eq!(CpuSet::from_words(&[0, 0]), CpuSet::new());
    }

    #[test]
    fn the_nth_cpu_of_an_intersection_is_found_without_building_it() {
        let a: CpuSet = [1, 3, 5, 64, 70, 130, 200].into_iter().collect();
        let b: CpuSet = [3, 4, 5, 64, 130, 131].into_iter().collect();
        let both = a.intersection(&b);
        assert_eq!(a.intersection_len(&b), both.len());
        for (n, cpu) in both.iter().enumerate() {
            assert_eq!(a.nth_of_intersection(&b, n), Some(cpu));
            assert_eq!(b.nth_of_intersection(&a, n), Some(cpu));
        }
        assert_eq!(a.nth_of_intersection(&b, both.len()), None);
        assert_eq!(a.nth_of_intersection(&CpuSet::new(), 0), None);
        assert_eq!(CpuSet::new().intersection_len(&a), 0);
    }

    #[test]
    fn iterates_in_ascending_order_across_words() {
        let set: CpuSet = [130, 0, 63, 64].into_iter().collect();
        assert_eq!(set.iter().collect::<Vec<_>>(), vec![0, 63, 64, 130]);
        assert_eq!(set.len(), 4);
        assert_eq!(set.span(), 131);
        assert_eq!(set.first(), Some(0));
        assert_eq!(set.last(), Some(130));
        assert_eq!(CpuSet::new().span(), 0);
        assert_eq!(CpuSet::new().iter().next(), None);
    }

    #[test]
    fn set_operations_return_new_canonical_sets() {
        let a = CpuSet::range(0, 70);
        let b = CpuSet::range(64, 200);
        assert_eq!(a.intersection(&b), CpuSet::range(64, 70));
        assert_eq!(a.union(&b), CpuSet::range(0, 200));
        assert_eq!(b.difference(&a), CpuSet::range(70, 200));
        assert_eq!(a.difference(&CpuSet::range(0, 70)), CpuSet::new());
        assert!(a.intersects(&b));
        assert!(CpuSet::range(65, 66).is_subset(&a));
        assert_eq!(a.with(500).without(500), a);
        assert_eq!(CpuSet::range(0, 1).without(0).words(), &[] as &[u64]);
    }

    #[test]
    fn formats_like_java_bitset() {
        assert_eq!(CpuSet::from_words(&[0b1101]).to_string(), "{0, 2, 3}");
        assert_eq!(CpuSet::new().to_string(), "{}");
    }
}
