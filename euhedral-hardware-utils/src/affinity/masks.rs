//! Little-endian logical-CPU masks (upstream `AffinityMasks`): word `i`, bit `b` is CPU `64 * i + b`.

use crate::util::CpuSet;

/// The largest logical CPU span a controller accepts.
pub const MAX_BITS: usize = 1_048_576;

/// The mask trimmed of trailing zero words, or `None` when it is empty, reaches `span`, or names a
/// CPU missing from `supported` (a hole in a sparse topology).
pub fn canonical(source: &[u64], span: usize, supported: &CpuSet) -> Option<Vec<u64>> {
    let length = source.iter().rposition(|&word| word != 0)? + 1;
    let words = &source[..length];
    let highest = (length - 1) * 64 + (63 - words[length - 1].leading_zeros() as usize);
    if highest >= span {
        return None;
    }
    let mask = CpuSet::from_words(words);
    mask.is_subset(supported).then(|| words.to_vec())
}

/// The number of words with at least one CPU; Windows rejects masks spanning more than one
/// processor group.
pub fn nonzero_words(mask: &[u64]) -> usize {
    mask.iter().filter(|&&word| word != 0).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_and_validates() {
        let supported: CpuSet = [0, 1, 2, 3, 65].into_iter().collect();
        assert_eq!(
            canonical(&[0b101, 0, 0, 0], 66, &supported),
            Some(vec![0b101])
        );
        assert_eq!(canonical(&[0, 0b10], 66, &supported), Some(vec![0, 0b10]));
        assert_eq!(canonical(&[], 66, &supported), None);
        assert_eq!(canonical(&[0, 0], 66, &supported), None);
        // CPU 64 is a hole; CPU 66 is outside the span.
        assert_eq!(canonical(&[0, 0b1], 66, &supported), None);
        assert_eq!(canonical(&[0, 0b100], 66, &supported), None);
    }

    #[test]
    fn counts_nonzero_words() {
        assert_eq!(nonzero_words(&[1, 0, 4]), 2);
        assert_eq!(nonzero_words(&[]), 0);
    }
}
