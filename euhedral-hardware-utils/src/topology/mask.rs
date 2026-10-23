//! Kernel-style hex CPU masks (upstream `MaskCodec`, `SystemInfo.fromHexMask`/`toHexMask`).

use crate::util::{CpuSet, cpu_set};
use std::error::Error;
use std::fmt;

/// Upper bound on the number of comma-separated chunks, so a hostile string cannot ask for a mask of
/// billions of bits.
pub const MAX_CHUNKS: usize = 1 << 26;

/// Why a hex mask did not parse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaskError {
    /// More than [`MAX_CHUNKS`] chunks.
    TooManyChunks(usize),
    /// Chunk `index`, counted from the left, is empty, not hex, or has more than 8 significant digits.
    InvalidChunk { index: usize, chunk: String },
}

impl fmt::Display for MaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaskError::TooManyChunks(count) => {
                write!(
                    f,
                    "mask has {count} chunks; at most {MAX_CHUNKS} (2^26) are supported"
                )
            }
            MaskError::InvalidChunk { index, chunk } => {
                write!(f, "invalid hex mask chunk at index {index}: \"{chunk}\"")
            }
        }
    }
}

impl Error for MaskError {}

/// Parses a kernel-style hex mask: comma-separated 32-bit chunks, most significant chunk first, so
/// chunk `k` from the right holds CPUs `32k ..= 32k + 31`.
///
/// Grammar: trailing empty chunks are ignored (`"1,"` is `{0}` and `","` is
/// empty) but a leading or middle empty chunk is an error; each chunk may carry surrounding
/// whitespace, one optional `+` and one optional `0x`/`0X` prefix, any number of leading zeros, and
/// at most 8 significant hex digits. A wider chunk is rejected.
pub fn parse_hex_mask(mask: &str) -> Result<CpuSet, MaskError> {
    let bytes = mask.as_bytes();
    if !bytes.contains(&b',') {
        let mut words = Vec::new();
        parse_chunk(mask, 0, bytes.len(), 0, 0, &mut words)?;
        return Ok(CpuSet::from_owned_words(words));
    }
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1] == b',' {
        end -= 1;
    }
    if end == 0 {
        return Ok(CpuSet::new());
    }
    let chunk_count = 1 + bytes[..end].iter().filter(|&&b| b == b',').count();
    if chunk_count > MAX_CHUNKS {
        return Err(MaskError::TooManyChunks(chunk_count));
    }
    let mut words = Vec::new();
    let mut start = 0;
    for index in 0..chunk_count {
        let next = bytes[start..end]
            .iter()
            .position(|&b| b == b',')
            .map_or(end, |offset| start + offset);
        parse_chunk(
            mask,
            start,
            next,
            index,
            chunk_count - 1 - index,
            &mut words,
        )?;
        start = next + 1;
    }
    Ok(CpuSet::from_owned_words(words))
}

fn parse_chunk(
    mask: &str,
    from: usize,
    to: usize,
    index: usize,
    from_right: usize,
    words: &mut Vec<u64>,
) -> Result<(), MaskError> {
    let bytes = mask.as_bytes();
    let invalid = || MaskError::InvalidChunk {
        index,
        chunk: mask[from..to].to_string(),
    };
    let is_space = |b: u8| b <= b' ';
    let (mut s, mut e) = (from, to);
    while s < e && is_space(bytes[s]) {
        s += 1;
    }
    while e > s && is_space(bytes[e - 1]) {
        e -= 1;
    }
    let mut sign_seen = false;
    if s < e && bytes[s] == b'+' {
        sign_seen = true;
        s += 1;
    }
    if e - s >= 2 && bytes[s] == b'0' && (bytes[s + 1] == b'x' || bytes[s + 1] == b'X') {
        s += 2;
        while s < e && is_space(bytes[s]) {
            s += 1;
        }
        if !sign_seen && s < e && bytes[s] == b'+' {
            s += 1;
        }
    }
    if s == e {
        return Err(invalid());
    }
    while s < e && bytes[s] == b'0' {
        s += 1;
    }
    if e - s > 8 {
        return Err(invalid());
    }
    let mut value: u64 = 0;
    for &b in &bytes[s..e] {
        let digit = (b as char).to_digit(16).ok_or_else(invalid)?;
        value = (value << 4) | u64::from(digit);
    }
    let base = from_right * 32;
    while value != 0 {
        cpu_set::set_bit(words, base + value.trailing_zeros() as usize);
        value &= value - 1;
    }
    Ok(())
}

/// Formats a set as a kernel-style hex mask: `"0"` when empty, otherwise the highest chunk unpadded
/// and every lower chunk as 8 hex digits, for example `"1,00000000,00000001"` for `{0, 64}`.
pub fn to_hex_mask(set: &CpuSet) -> String {
    let words = set.words();
    if words.is_empty() {
        return "0".to_string();
    }
    let mut chunks: Vec<String> = Vec::with_capacity(words.len() * 2);
    for &word in words.iter().rev() {
        for half in [(word >> 32) as u32, word as u32] {
            if chunks.is_empty() {
                if half != 0 {
                    chunks.push(format!("{half:x}"));
                }
            } else {
                chunks.push(format!("{half:08x}"));
            }
        }
    }
    chunks.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(ids: &[usize]) -> CpuSet {
        ids.iter().copied().collect()
    }

    fn parse(text: &str) -> CpuSet {
        parse_hex_mask(text).unwrap_or_else(|e| panic!("{text:?}: {e}"))
    }

    #[test]
    fn preserves_canonical_cpu_mask_text() {
        let cases: [(&[usize], &str); 6] = [
            (&[], "0"),
            (&[0], "1"),
            (&[31, 32], "1,80000000"),
            (&[0, 32, 63], "80000001,00000001"),
            (&[0, 64], "1,00000000,00000001"),
            (&[127], "80000000,00000000,00000000,00000000"),
        ];
        for (ids, text) in cases {
            assert_eq!(to_hex_mask(&bits(ids)), text);
            assert_eq!(parse(text), bits(ids), "{text}");
        }
        assert_eq!(parse("0x1"), bits(&[0]));
        assert!(parse_hex_mask("not-hex").is_err());
        assert!(parse_hex_mask("0x").is_err());
    }

    #[test]
    fn rejects_chunks_wider_than_32_bits_and_embedded_prefixes() {
        assert!(parse_hex_mask("ffffffffff").is_err());
        assert!(parse_hex_mask("ff0x01").is_err());
        assert_eq!(parse("0X3"), bits(&[0, 1]));
        assert_eq!(parse("1,00000000"), bits(&[32]));
    }

    #[test]
    fn round_trips_sparse_high_bits() {
        let set = bits(&[0, 63, 64, 1023, 1024]);
        assert_eq!(parse(&to_hex_mask(&set)), set);
    }

    #[test]
    fn accepts_lenient_inputs() {
        let cases: [(&str, &[usize]); 15] = [
            ("1,", &[0]),
            (",", &[]),
            ("1,,,", &[0]),
            (",,", &[]),
            ("0000000001", &[0]),
            ("0000000000000000000000000001", &[0]),
            ("0000", &[]),
            ("+1", &[0]),
            ("+0x1", &[0]),
            ("0x+1", &[0]),
            ("0x 1", &[0]),
            (" 1 ", &[0]),
            ("\t0x\t3\t", &[0, 1]),
            ("1,00000000", &[32]),
            ("00000000,00000001", &[0]),
        ];
        for (text, ids) in cases {
            assert_eq!(parse(text), bits(ids), "{text:?}");
        }
    }

    #[test]
    fn rejects_misparsed_and_malformed_inputs() {
        for text in [
            "1,,2",
            ",1",
            "00x1",
            "0x0x1",
            "+ 1",
            "1, ",
            "0x",
            "+",
            "100000000",
            "",
            "é",
        ] {
            assert!(parse_hex_mask(text).is_err(), "{text:?}");
        }
        let error = parse_hex_mask("1,zz").unwrap_err();
        assert_eq!(
            error,
            MaskError::InvalidChunk {
                index: 1,
                chunk: "zz".to_string()
            }
        );
    }
}
