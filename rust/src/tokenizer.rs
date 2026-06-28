//! Byte-level BPE encode/decode.
//!
//! Tokens are byte sequences (`Vec<u8>`), so decoding is *exact* on arbitrary
//! binary input: the base alphabet is all 256 single bytes (no UNK), and merges
//! only ever concatenate adjacent token byte-strings. `decode(encode(x)) == x`
//! always holds.
//!
//! The DEFAULT build constructs a `BpeModel` in memory (used by tests). Loading
//! the notebook's exported `tokenizer.json` is done under the `neural` feature
//! (serde_json); see docs/rust_integration_blueprint.md for the exact JSON
//! schema (GPT-2 byte<->unicode map + rank-ordered merges).

use std::collections::HashMap;

pub struct BpeModel {
    /// id -> the literal bytes that token expands to.
    id_to_bytes: Vec<Vec<u8>>,
    /// token bytes -> id.
    bytes_to_id: HashMap<Vec<u8>, u32>,
    /// (left, right) token bytes -> merge rank (lower = applied first).
    merge_rank: HashMap<(Vec<u8>, Vec<u8>), u32>,
}

impl BpeModel {
    /// A model whose base alphabet is all 256 bytes. `merges` are pairs of byte
    /// strings in priority order (rank = index). Each merge also registers the
    /// concatenated token in the vocabulary.
    pub fn new(merges: Vec<(Vec<u8>, Vec<u8>)>) -> Self {
        let mut id_to_bytes: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        let mut bytes_to_id: HashMap<Vec<u8>, u32> = HashMap::new();
        for (i, t) in id_to_bytes.iter().enumerate() {
            bytes_to_id.insert(t.clone(), i as u32);
        }
        let mut merge_rank = HashMap::new();
        for (rank, (l, r)) in merges.into_iter().enumerate() {
            let mut merged = l.clone();
            merged.extend_from_slice(&r);
            if !bytes_to_id.contains_key(&merged) {
                bytes_to_id.insert(merged.clone(), id_to_bytes.len() as u32);
                id_to_bytes.push(merged);
            }
            merge_rank.insert((l, r), rank as u32);
        }
        BpeModel { id_to_bytes, bytes_to_id, merge_rank }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Greedy BPE: repeatedly merge the adjacent pair with the lowest rank.
    pub fn encode(&self, bytes: &[u8]) -> Vec<u32> {
        if bytes.is_empty() {
            return Vec::new();
        }
        let mut tokens: Vec<Vec<u8>> = bytes.iter().map(|&b| vec![b]).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..tokens.len() - 1 {
                if let Some(&rank) = self.merge_rank.get(&(tokens[i].clone(), tokens[i + 1].clone()))
                {
                    if best.map_or(true, |(br, _)| rank < br) {
                        best = Some((rank, i));
                    }
                }
            }
            match best {
                None => break,
                Some((_, i)) => {
                    let mut merged = tokens[i].clone();
                    merged.extend_from_slice(&tokens[i + 1]);
                    tokens[i] = merged;
                    tokens.remove(i + 1);
                }
            }
        }
        tokens
            .into_iter()
            .map(|t| *self.bytes_to_id.get(&t).expect("token must be in vocab"))
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            out.extend_from_slice(&self.id_to_bytes[id as usize]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_model() -> BpeModel {
        // A few merges over ASCII; round-trip must hold regardless.
        BpeModel::new(vec![
            (b"a".to_vec(), b"b".to_vec()),   // ab
            (b"ab".to_vec(), b"c".to_vec()),  // abc
            (b"l".to_vec(), b"o".to_vec()),   // lo
        ])
    }

    #[test]
    fn roundtrip_is_exact_on_arbitrary_bytes() {
        let m = toy_model();
        let cases: Vec<&[u8]> = vec![
            b"",
            b"abc",
            b"abcabc lo lo",
            b"\x00\xff\x80 binary\x01",
            &[0u8, 1, 2, 254, 255],
        ];
        for c in cases {
            let ids = m.encode(c);
            assert_eq!(m.decode(&ids), c, "roundtrip failed for {:?}", c);
        }
    }

    #[test]
    fn merges_actually_apply() {
        let m = toy_model();
        // "abc" should collapse to a single token via ab -> abc.
        assert_eq!(m.encode(b"abc").len(), 1);
        // "lo" collapses to one token, "x" stays separate.
        assert_eq!(m.encode(b"lox").len(), 2);
    }
}
