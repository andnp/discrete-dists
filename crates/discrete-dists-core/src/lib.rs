use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::fmt;
use std::iter;

mod rng;

pub use rng::Rng;

// Renormalize when the multiplier leaves this range, so stored weights stay far from overflow and underflow.
const SCALE_MIN: f64 = 1e-100;
const SCALE_MAX: f64 = 1e100;

#[derive(Serialize, Deserialize, Clone)]
#[serde(try_from = "TreeWire", into = "TreeWire")]
pub struct Tree {
    size: u32,
    total_size: u32,
    // Implicit binary heap: root at 1, children of i at 2i and 2i+1, leaf k at total_size + k.
    data: Vec<f64>,
    // Multiplier applied to stored weights; kept out of TreeWire so pickles keep the historical format.
    scale: f64,
}

// Field order and types are the pickle format; bincode encodes them positionally.
#[derive(Serialize, Deserialize)]
struct TreeWire {
    size: u32,
    total_size: u32,
    raw: Vec<Array1<f64>>,
}

impl From<Tree> for TreeWire {
    fn from(mut tree: Tree) -> Self {
        if tree.scale != 1.0 {
            tree.renormalize();
        }

        let total = tree.total_size as usize;
        let raw = (0..)
            .map(|r| total >> r)
            .take_while(|&len| len > 0)
            .map(|len| Array1::from_vec(tree.data[len..2 * len].to_vec()))
            .collect();

        TreeWire {
            size: tree.size,
            total_size: tree.total_size,
            raw,
        }
    }
}

impl TryFrom<TreeWire> for Tree {
    type Error = String;

    fn try_from(wire: TreeWire) -> Result<Self, Self::Error> {
        if wire.total_size == 0 {
            if wire.size != 0 || !wire.raw.is_empty() {
                return Err(format!(
                    "empty tree must have size 0 and no levels, got size {} and {} levels",
                    wire.size,
                    wire.raw.len()
                ));
            }
        } else {
            if !wire.total_size.is_power_of_two() {
                return Err(format!(
                    "total_size must be a power of two, got {}",
                    wire.total_size
                ));
            }
            if wire.size > wire.total_size {
                return Err(format!(
                    "size {} exceeds total_size {}",
                    wire.size, wire.total_size
                ));
            }
            let n_layers = wire.total_size.ilog2() as usize + 1;
            if wire.raw.len() != n_layers {
                return Err(format!(
                    "expected {n_layers} levels for total_size {}, got {}",
                    wire.total_size,
                    wire.raw.len()
                ));
            }
            for (r, layer) in wire.raw.iter().enumerate() {
                let expected = (wire.total_size >> r) as usize;
                if layer.len() != expected {
                    return Err(format!(
                        "level {r} must have {expected} nodes, got {}",
                        layer.len()
                    ));
                }
            }
        }

        let total = wire.total_size as usize;
        let mut data = vec![0.0; 2 * total];
        for (r, layer) in wire.raw.iter().enumerate() {
            let len = total >> r;
            for (dst, src) in data[len..2 * len].iter_mut().zip(layer.iter()) {
                *dst = *src;
            }
        }

        Ok(Tree {
            size: wire.size,
            total_size: wire.total_size,
            data,
            scale: 1.0,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TreeError {
    IndexOutOfBounds { idx: i64, size: u32 },
    NonFiniteWeight { idx: i64, value: f64 },
    NegativeWeight { idx: i64, value: f64 },
    LengthMismatch { idxs: usize, values: usize },
    InvalidScale { factor: f64 },
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeError::IndexOutOfBounds { idx, size } => {
                write!(f, "index {idx} out of bounds for SumTree of size {size}")
            }
            TreeError::NonFiniteWeight { idx, value } => {
                write!(f, "value at index {idx} must be finite, got {value}")
            }
            TreeError::NegativeWeight { idx, value } => {
                write!(f, "value at index {idx} must be nonnegative, got {value}")
            }
            TreeError::LengthMismatch { idxs, values } => write!(
                f,
                "idxs and values must have the same length, got {idxs} and {values}"
            ),
            TreeError::InvalidScale { factor } => {
                write!(f, "scale factor must be finite and positive, got {factor}")
            }
        }
    }
}

impl std::error::Error for TreeError {}

impl Tree {
    pub fn new(size: u32) -> Self {
        let total_size = u32::next_power_of_two(size);

        Tree {
            size,
            total_size,
            data: vec![0.0; 2 * total_size as usize],
            scale: 1.0,
        }
    }

    pub fn empty() -> Self {
        Tree {
            size: 0,
            total_size: 0,
            data: vec![],
            scale: 1.0,
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    fn checked_index(&self, idx: i64) -> Result<usize, TreeError> {
        if idx < 0 || idx >= self.size as i64 {
            return Err(TreeError::IndexOutOfBounds {
                idx,
                size: self.size,
            });
        }

        Ok(idx as usize)
    }

    fn validate_weight(idx: i64, value: f64) -> Result<(), TreeError> {
        if !value.is_finite() {
            return Err(TreeError::NonFiniteWeight { idx, value });
        }

        if value < 0.0 {
            return Err(TreeError::NegativeWeight { idx, value });
        }

        Ok(())
    }

    fn set_validated_value(&mut self, checked_idx: usize, weight: f64) {
        let mut value = weight / self.scale;
        // A tiny multiplier can push a valid weight past f64 range; folding
        // the multiplier back in first keeps every stored value finite.
        if !value.is_finite() {
            self.renormalize();
            value = weight;
        }
        let len = self.data.len();
        let mut i = self.total_size as usize + checked_idx;
        debug_assert!(i < len);
        // SAFETY: checked_index guarantees checked_idx < size <= total_size, so i < 2 * total_size == len.
        unsafe { *self.data.get_unchecked_mut(i) = value };

        // Each parent is rebuilt as node + sibling rather than accumulating
        // deltas, so sums never drift; the new node value is already known,
        // so only the sibling needs reading.
        let mut node = value;
        while i > 1 {
            debug_assert!(i ^ 1 < len);
            // SAFETY: i < len and len is even, so flipping the low bit keeps i ^ 1 < len.
            node += unsafe { *self.data.get_unchecked(i ^ 1) };
            i >>= 1;
            debug_assert!(i < len);
            // SAFETY: i was halved from a value below len and is at least 1, so it is in bounds.
            unsafe { *self.data.get_unchecked_mut(i) = node };
        }
    }

    // Slow path for query: never steps into a zero-weight subtree, so it
    // lands on a positive leaf whenever the tree has positive total.
    fn descend_positive(&self, value: f64) -> i64 {
        let total_size = self.total_size as usize;
        let mut node = 1usize;
        let mut total = 0.0;
        while node < total_size {
            let idx = 2 * node;
            let left = self.data[idx];
            let right = self.data[idx + 1];
            if right > 0.0 && (left == 0.0 || value - total >= left) {
                total += left;
                node = idx + 1;
            } else {
                node = idx;
            }
        }
        (node - total_size) as i64
    }

    pub fn update(&mut self, idxs: &[i64], values: &[f64]) -> Result<(), TreeError> {
        if idxs.len() != values.len() {
            return Err(TreeError::LengthMismatch {
                idxs: idxs.len(),
                values: values.len(),
            });
        }

        let validated = iter::zip(idxs, values)
            .map(|(idx, value)| {
                let checked_idx = self.checked_index(*idx)?;
                Self::validate_weight(*idx, *value)?;
                Ok((checked_idx, *value))
            })
            .collect::<Result<Vec<_>, TreeError>>()?;

        for (checked_idx, value) in validated {
            self.set_validated_value(checked_idx, value);
        }

        Ok(())
    }

    pub fn update_single(&mut self, idx: i64, value: f64) -> Result<(), TreeError> {
        Self::validate_weight(idx, value)?;
        let checked_idx = self.checked_index(idx)?;
        self.set_validated_value(checked_idx, value);

        Ok(())
    }

    pub fn get_value(&self, idx: i64) -> Result<f64, TreeError> {
        Ok(self.data[self.total_size as usize + self.checked_index(idx)?] * self.scale)
    }

    pub fn get_values(&self, idxs: &[i64]) -> Result<Vec<f64>, TreeError> {
        let leaf_offset = self.total_size as usize;
        let idxs: Vec<usize> = idxs
            .iter()
            .map(|idx| self.checked_index(*idx))
            .collect::<Result<_, _>>()?;

        Ok(idxs
            .iter()
            .map(|&i| self.data[leaf_offset + i] * self.scale)
            .collect())
    }

    pub fn total(&self) -> f64 {
        self.data.get(1).copied().unwrap_or(0.0) * self.scale
    }

    pub fn scale(&mut self, factor: f64) -> Result<(), TreeError> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(TreeError::InvalidScale { factor });
        }

        self.scale *= factor;
        if !(SCALE_MIN..=SCALE_MAX).contains(&self.scale) {
            self.renormalize();
        }

        Ok(())
    }

    fn renormalize(&mut self) {
        let total = self.total_size as usize;
        for leaf in &mut self.data[total..] {
            *leaf *= self.scale;
        }
        self.scale = 1.0;
        self.rebuild();
    }

    fn rebuild(&mut self) {
        let total = self.total_size as usize;
        for i in (1..total).rev() {
            self.data[i] = self.data[2 * i] + self.data[2 * i + 1];
        }
    }

    // `out` is the descent accumulator: it holds heap node indices, reset to the root before use.
    pub fn query(&self, values: &[f64], out: &mut [i64]) {
        assert_eq!(values.len(), out.len());
        let n = values.len();
        let total_size = self.total_size as usize;
        let depth = total_size.max(1).trailing_zeros();
        // Query values arrive in true-weight units; the heap stores them
        // divided by the multiplier.
        let mut remaining: Vec<f64> = values.iter().map(|v| v / self.scale).collect();

        out.fill(1);

        for _ in 0..depth {
            for j in 0..n {
                let node = out[j] as usize;
                debug_assert!(node < total_size);
                let idx = 2 * node;
                debug_assert!(idx + 1 < self.data.len());
                // SAFETY: node is an internal node (the walk starts at the root and descends depth levels), so node < total_size and idx + 1 < 2 * total_size == data.len().
                let left = unsafe { *self.data.get_unchecked(idx) };
                let m = remaining[j] >= left;
                remaining[j] -= if m { left } else { 0. };
                out[j] = (idx + m as usize) as i64;
            }
        }

        for j in 0..n {
            let leaf = out[j] as usize - total_size;
            out[j] = if self.data[total_size + leaf] <= 0.0 {
                self.descend_positive(values[j] / self.scale)
            } else {
                leaf as i64
            };
        }

        for idx in out.iter_mut() {
            *idx = min(*idx, (self.size - 1) as i64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;

    #[derive(Debug, Clone)]
    enum Op {
        Single(i64, f64),
        Batch(Vec<(i64, f64)>),
    }

    fn weight() -> impl Strategy<Value = f64> {
        prop_oneof![
            2 => Just(0.0),
            1 => Just(1e-9),
            3 => 0.0..=1e6f64,
        ]
    }

    fn pair(size: u32) -> impl Strategy<Value = (i64, f64)> {
        (0..size as i64, weight())
    }

    fn ops() -> impl Strategy<Value = (u32, Vec<Op>)> {
        (1u32..=2048).prop_flat_map(|size| {
            let op = prop_oneof![
                pair(size).prop_map(|(idx, w)| Op::Single(idx, w)),
                vec(pair(size), 1..=16).prop_map(Op::Batch),
            ];
            (Just(size), vec(op, 0..=200))
        })
    }

    fn apply(tree: &mut Tree, op: &Op) {
        match op {
            Op::Single(idx, w) => tree.update_single(*idx, *w).unwrap(),
            Op::Batch(pairs) => {
                let (idxs, values): (Vec<i64>, Vec<f64>) = pairs.iter().copied().unzip();
                tree.update(&idxs, &values).unwrap();
            }
        }
    }

    fn build(size: u32, ops: &[Op]) -> (Tree, Vec<f64>) {
        let mut tree = Tree::new(size);
        let mut leaves = vec![0.0; tree.total_size as usize];
        for op in ops {
            apply(&mut tree, op);
            match op {
                Op::Single(idx, w) => leaves[*idx as usize] = *w,
                Op::Batch(pairs) => {
                    for (idx, w) in pairs {
                        leaves[*idx as usize] = *w;
                    }
                }
            }
        }
        (tree, leaves)
    }

    fn leaf_slice(tree: &Tree) -> &[f64] {
        let total = tree.total_size as usize;
        &tree.data[total..2 * total]
    }

    #[test]
    fn wire_round_trip_preserves_levels() {
        let mut tree = Tree::new(5);
        tree.update(&[0, 3, 4], &[1.5, 0.0, 2.0]).unwrap();
        let before: Vec<Vec<f64>> = TreeWire::from(tree.clone())
            .raw
            .iter()
            .map(|l| l.to_vec())
            .collect();

        let back = TreeWire::from(Tree::try_from(TreeWire::from(tree)).unwrap());

        let after: Vec<Vec<f64>> = back.raw.iter().map(|l| l.to_vec()).collect();
        assert_eq!(after, before);
        assert_eq!(back.size, 5);
        assert_eq!(back.total_size, 8);
    }

    #[test]
    fn wire_rejects_level_with_wrong_length() {
        let mut wire = TreeWire::from(Tree::new(4));
        wire.raw[1] = Array1::zeros(3);

        assert!(Tree::try_from(wire).is_err());
    }

    #[test]
    fn scale_rejects_non_finite_or_nonpositive_factors_and_changes_nothing() {
        let mut tree = Tree::new(3);
        tree.update(&[0, 2], &[1.0, 3.0]).unwrap();

        for factor in [0.0, -2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = tree.scale(factor).unwrap_err();
            assert!(
                matches!(err, TreeError::InvalidScale { .. }),
                "factor {factor}: got {err:?}"
            );
            assert_eq!(tree.get_values(&[0, 1, 2]).unwrap(), vec![1.0, 0.0, 3.0]);
            assert_eq!(tree.total(), 4.0);
        }
    }

    #[test]
    fn update_after_heavy_decay_stays_finite() {
        let mut tree = Tree::new(4);
        tree.update(&[0, 1], &[1.0, 2.0]).unwrap();
        tree.scale(1e-90).unwrap();
        tree.update_single(2, 1e300).unwrap();

        assert_eq!(tree.get_value(2).unwrap(), 1e300);
        assert!(tree.total().is_finite());
        assert!((tree.get_value(1).unwrap() / 2e-90 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn scaled_tree_round_trips_weights_and_resets_scale() {
        let mut tree = Tree::new(5);
        tree.update(&[0, 3, 4], &[1.5, 0.0, 2.0]).unwrap();
        tree.scale(0.5).unwrap();
        tree.scale(0.1).unwrap();
        tree.update(&[1], &[4.0]).unwrap();

        let idxs = [0, 1, 2, 3, 4];
        let before = tree.get_values(&idxs).unwrap();
        let total_before = tree.total();

        let back = Tree::try_from(TreeWire::from(tree)).unwrap();

        assert_eq!(back.scale, 1.0);
        for (k, (after, expected)) in back
            .get_values(&idxs)
            .unwrap()
            .iter()
            .zip(&before)
            .enumerate()
        {
            assert!(
                (after - expected).abs() <= 1e-12 * expected.abs(),
                "leaf {k}: got {after}, expected {expected}"
            );
        }
        assert!((back.total() - total_before).abs() <= 1e-12 * total_before);
    }

    proptest! {
        #[test]
        fn internal_nodes_are_exact_child_sums_and_unscaled_leaves_hold_last_write(
            (size, ops) in ops(),
        ) {
            let (tree, leaves) = build(size, &ops);

            for i in 1..tree.total_size as usize {
                prop_assert_eq!(tree.data[i], tree.data[2 * i] + tree.data[2 * i + 1]);
            }

            prop_assert_eq!(tree.scale, 1.0);
            prop_assert_eq!(leaf_slice(&tree).to_vec(), leaves);
        }

        #[test]
        fn total_matches_sum_of_leaves(
            (size, ops) in ops(),
        ) {
            let (tree, _) = build(size, &ops);
            let total = tree.total();

            prop_assert_eq!(total, tree.data[1]);
            let sum: f64 = leaf_slice(&tree).iter().sum();
            let diff = (total - sum).abs();
            prop_assert!(diff <= 1e-12 || diff <= 1e-9 * sum.abs());
        }

        #[test]
        fn query_returns_in_range_positive_leaves(
            (size, ops) in ops(),
            fractions in vec(0.0..1.0f64, 0..=64),
        ) {
            let (tree, _) = build(size, &ops);
            let total = tree.total();
            prop_assume!(total > 0.0);

            let mut values: Vec<f64> = fractions.iter().map(|f| f * total).collect();
            values.push(0.0);
            values.push(total);
            let mut out = vec![0i64; values.len()];
            tree.query(&values, &mut out);

            for idx in out {
                prop_assert!((0..size as i64).contains(&idx));
                prop_assert!(tree.get_value(idx).unwrap() > 0.0);
            }
        }

        #[test]
        fn update_is_atomic_on_invalid_pair(
            (size, ops) in ops(),
            batch in vec((0..2048i64, weight()), 0..=16),
            position in any::<prop::sample::Index>(),
            defect in 0..4u8,
        ) {
            let (mut tree, _) = build(size, &ops);
            let before = leaf_slice(&tree).to_vec();

            let mut pairs: Vec<(i64, f64)> = batch
                .into_iter()
                .map(|(idx, w)| (idx % size as i64, w))
                .collect();
            let (bad_idx, bad_w) = match defect {
                0 => (size as i64 + 1, 1.0),
                1 => (-1, 1.0),
                2 => (0, f64::NAN),
                _ => (0, -1e-9),
            };
            pairs.insert(position.index(pairs.len() + 1), (bad_idx, bad_w));

            let (idxs, values): (Vec<i64>, Vec<f64>) = pairs.into_iter().unzip();
            let err = tree.update(&idxs, &values).unwrap_err();

            let kind_ok = match defect {
                0 | 1 => matches!(err, TreeError::IndexOutOfBounds { .. }),
                2 => matches!(err, TreeError::NonFiniteWeight { .. }),
                _ => matches!(err, TreeError::NegativeWeight { .. }),
            };
            prop_assert!(kind_ok, "unexpected error {err:?} for defect {defect}");
            prop_assert_eq!(leaf_slice(&tree).to_vec(), before);
        }
    }

    #[derive(Debug, Clone)]
    enum ScaleOp {
        Write(i64, f64),
        Scale(f64),
    }

    fn scale_factor() -> impl Strategy<Value = f64> {
        prop_oneof![
            8 => 0.5..=1.5f64,
            1 => Just(1e-60),
            1 => Just(1e60),
        ]
    }

    fn scale_ops(size: u32) -> impl Strategy<Value = Vec<ScaleOp>> {
        let op = prop_oneof![
            3 => pair(size).prop_map(|(idx, w)| ScaleOp::Write(idx, w)),
            1 => scale_factor().prop_map(ScaleOp::Scale),
        ];
        vec(op, 0..=200)
    }

    fn approx_eq(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() <= (1e-9 * expected.abs()).max(1e-300)
    }

    proptest! {
        #[test]
        fn scaled_weights_match_model_and_query_stays_positive(
            (size, ops) in (1u32..=64).prop_flat_map(|size| (Just(size), scale_ops(size))),
        ) {
            let mut tree = Tree::new(size);
            let mut model = vec![0.0; size as usize];
            for op in &ops {
                match op {
                    ScaleOp::Write(idx, w) => {
                        tree.update_single(*idx, *w).unwrap();
                        model[*idx as usize] = *w;
                    }
                    ScaleOp::Scale(factor) => {
                        tree.scale(*factor).unwrap();
                        let before = model.clone();
                        for w in &mut model {
                            *w *= factor;
                        }
                        // Past the normal f64 range the true weight has no faithful value to compare against.
                        prop_assume!(before.iter().zip(&model).all(|(b, a)| *b == 0.0 || a.is_normal()));
                    }
                }
            }

            for (k, expected) in model.iter().enumerate() {
                let actual = tree.get_value(k as i64).unwrap();
                prop_assert!(approx_eq(actual, *expected), "leaf {k}: got {actual}, expected {expected}");
            }
            let expected_total: f64 = model.iter().sum();
            prop_assert!(approx_eq(tree.total(), expected_total), "total: got {}, expected {expected_total}", tree.total());

            let total = tree.total();
            if total > 0.0 {
                let values: Vec<f64> = (0..=8).map(|i| total * i as f64 / 8.0).collect();
                let mut out = vec![0i64; values.len()];
                tree.query(&values, &mut out);
                for idx in out {
                    prop_assert!((0..size as i64).contains(&idx));
                    prop_assert!(model[idx as usize] > 0.0);
                }
            }
        }
    }
}
