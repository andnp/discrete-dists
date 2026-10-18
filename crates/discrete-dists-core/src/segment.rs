use crate::TreeError;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

pub trait Combine {
    const IDENTITY: f64;
    fn combine(a: f64, b: f64) -> f64;
}

#[derive(Clone, Copy, Debug)]
pub struct Min;

impl Combine for Min {
    const IDENTITY: f64 = f64::INFINITY;

    fn combine(a: f64, b: f64) -> f64 {
        f64::min(a, b)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Max;

impl Combine for Max {
    const IDENTITY: f64 = f64::NEG_INFINITY;

    fn combine(a: f64, b: f64) -> f64 {
        f64::max(a, b)
    }
}

// Same implicit heap layout as `Tree`. Min and Max cannot be undone by subtraction, so parents are always recomputed from both children.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SegmentTree<C> {
    size: u32,
    total_size: u32,
    data: Vec<f64>,
    #[serde(skip)]
    _combine: PhantomData<C>,
}

impl<C: Combine> SegmentTree<C> {
    pub fn new(size: u32) -> Self {
        let total_size = u32::next_power_of_two(size);

        SegmentTree {
            size,
            total_size,
            data: vec![C::IDENTITY; 2 * total_size as usize],
            _combine: PhantomData,
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn root(&self) -> f64 {
        self.data[1]
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

    // NaN is rejected because `f64::min` and `f64::max` skip it, which would leave the root disagreeing with the leaves.
    fn validate_value(idx: i64, value: f64) -> Result<(), TreeError> {
        if value.is_nan() {
            return Err(TreeError::NonFiniteWeight { idx, value });
        }

        Ok(())
    }

    fn set_raw(&mut self, checked_idx: usize, value: f64) {
        let mut i = self.total_size as usize + checked_idx;
        self.data[i] = value;
        let mut node = value;
        while i > 1 {
            node = C::combine(node, self.data[i ^ 1]);
            i >>= 1;
            // Ancestors depend only on their children, so an unchanged node leaves the rest of the path unchanged.
            if self.data[i] == node {
                break;
            }
            self.data[i] = node;
        }
    }

    pub fn get(&self, idx: i64) -> Result<f64, TreeError> {
        Ok(self.data[self.total_size as usize + self.checked_index(idx)?])
    }

    pub fn set(&mut self, idx: i64, value: f64) -> Result<(), TreeError> {
        Self::validate_value(idx, value)?;
        let checked_idx = self.checked_index(idx)?;
        self.set_raw(checked_idx, value);

        Ok(())
    }

    pub fn update(&mut self, idxs: &[i64], values: &[f64]) -> Result<(), TreeError> {
        if idxs.len() != values.len() {
            return Err(TreeError::LengthMismatch {
                idxs: idxs.len(),
                values: values.len(),
            });
        }

        let validated = std::iter::zip(idxs, values)
            .map(|(idx, value)| {
                Self::validate_value(*idx, *value)?;
                Ok((self.checked_index(*idx)?, *value))
            })
            .collect::<Result<Vec<_>, TreeError>>()?;

        for (checked_idx, value) in validated {
            self.set_raw(checked_idx, value);
        }

        Ok(())
    }

    pub fn reset(&mut self, idx: i64) -> Result<(), TreeError> {
        let checked_idx = self.checked_index(idx)?;
        self.set_raw(checked_idx, C::IDENTITY);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;

    #[derive(Debug, Clone)]
    enum Op {
        Set(usize, f64),
        Reset(usize),
    }

    fn value() -> impl Strategy<Value = f64> {
        prop_oneof![
            1 => Just(f64::INFINITY),
            1 => Just(f64::NEG_INFINITY),
            4 => -1e6..=1e6f64,
        ]
    }

    fn op(size: u32) -> impl Strategy<Value = Op> {
        prop_oneof![
            3 => (0..size as usize, value()).prop_map(|(idx, v)| Op::Set(idx, v)),
            1 => (0..size as usize).prop_map(Op::Reset),
        ]
    }

    fn case() -> impl Strategy<Value = (u32, Vec<Op>)> {
        (1u32..=300).prop_flat_map(|size| (Just(size), vec(op(size), 0..=200)))
    }

    fn fold<C: Combine>(values: &[f64]) -> f64 {
        values.iter().fold(C::IDENTITY, |a, &b| C::combine(a, b))
    }

    fn check_against_brute_force<C: Combine>(size: u32, ops: &[Op]) {
        let mut tree = SegmentTree::<C>::new(size);
        let mut model = vec![C::IDENTITY; size as usize];

        for op in ops {
            match *op {
                Op::Set(idx, v) => {
                    tree.set(idx as i64, v).unwrap();
                    model[idx] = v;
                }
                Op::Reset(idx) => {
                    tree.reset(idx as i64).unwrap();
                    model[idx] = C::IDENTITY;
                }
            }

            assert_eq!(tree.root(), fold::<C>(&model));
        }

        for (idx, v) in model.iter().enumerate() {
            assert_eq!(tree.get(idx as i64).unwrap(), *v);
        }
    }

    proptest! {
        #[test]
        fn min_root_matches_fold_over_leaves((size, ops) in case()) {
            check_against_brute_force::<Min>(size, &ops);
        }

        #[test]
        fn max_root_matches_fold_over_leaves((size, ops) in case()) {
            check_against_brute_force::<Max>(size, &ops);
        }
    }

    #[test]
    fn empty_tree_root_is_identity() {
        assert_eq!(SegmentTree::<Min>::new(4).root(), f64::INFINITY);
        assert_eq!(SegmentTree::<Max>::new(4).root(), f64::NEG_INFINITY);
    }

    #[test]
    fn size_zero_tree_has_identity_root_and_rejects_every_index() {
        let mut tree = SegmentTree::<Min>::new(0);

        assert_eq!(tree.size(), 0);
        assert_eq!(tree.root(), f64::INFINITY);
        assert_eq!(
            tree.get(0),
            Err(TreeError::IndexOutOfBounds { idx: 0, size: 0 })
        );
        assert_eq!(
            tree.reset(0),
            Err(TreeError::IndexOutOfBounds { idx: 0, size: 0 })
        );
    }

    #[test]
    fn set_rejects_nan_and_out_of_bounds_indices_without_changing_tree() {
        let mut tree = SegmentTree::<Max>::new(3);
        tree.set(1, 2.0).unwrap();

        assert!(matches!(
            tree.set(1, f64::NAN),
            Err(TreeError::NonFiniteWeight { idx: 1, .. })
        ));
        assert_eq!(
            tree.set(3, 1.0),
            Err(TreeError::IndexOutOfBounds { idx: 3, size: 3 })
        );
        assert_eq!(tree.get(1), Ok(2.0));
        assert_eq!(tree.root(), 2.0);
    }

    #[test]
    fn update_rejects_length_mismatch() {
        let mut tree = SegmentTree::<Min>::new(4);

        assert_eq!(
            tree.update(&[0, 1], &[1.0]),
            Err(TreeError::LengthMismatch { idxs: 2, values: 1 })
        );
    }

    #[test]
    fn update_is_atomic_on_invalid_pair() {
        let mut tree = SegmentTree::<Min>::new(4);
        tree.update(&[0, 1], &[5.0, 6.0]).unwrap();

        assert!(matches!(
            tree.update(&[2, 3, 9], &[1.0, f64::NAN, 1.0]),
            Err(TreeError::NonFiniteWeight { idx: 3, .. })
        ));
        assert_eq!(
            tree.update(&[2, 9], &[1.0, 1.0]),
            Err(TreeError::IndexOutOfBounds { idx: 9, size: 4 })
        );
        assert_eq!(tree.get(2), Ok(f64::INFINITY));
        assert_eq!(tree.root(), 5.0);
    }

    #[test]
    fn bincode_round_trip_preserves_root_and_leaves() {
        let config = bincode::config::standard();
        let mut tree = SegmentTree::<Min>::new(5);
        tree.update(&[0, 4], &[2.0, -1.0]).unwrap();

        let bytes = bincode::serde::encode_to_vec(&tree, config).unwrap();
        let (restored, _): (SegmentTree<Min>, _) =
            bincode::serde::decode_from_slice(&bytes, config).unwrap();

        assert_eq!(restored.size(), 5);
        assert_eq!(restored.root(), -1.0);
        assert_eq!(restored.get(0), Ok(2.0));
        assert_eq!(restored.get(1), Ok(f64::INFINITY));
    }
}
