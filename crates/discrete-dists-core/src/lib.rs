use ndarray::{Array1, Axis};
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::fmt;
use std::iter;

// Field order and types are the pickle format; bincode encodes them positionally.
#[derive(Serialize, Deserialize, Clone)]
pub struct Tree {
    size: u32,
    total_size: u32,
    raw: Vec<Array1<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TreeError {
    IndexOutOfBounds { idx: i64, size: u32 },
    NonFiniteWeight { idx: i64, value: f64 },
    NegativeWeight { idx: i64, value: f64 },
    LengthMismatch { idxs: usize, values: usize },
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
        }
    }
}

impl std::error::Error for TreeError {}

impl Tree {
    pub fn new(size: u32) -> Self {
        let total_size = u32::next_power_of_two(size);
        let n_layers = u32::ilog2(total_size) + 1;

        let dummy = Array1::<f64>::zeros(1);
        let mut layers = vec![dummy; n_layers as usize];

        for i in (0..n_layers).rev() {
            let r = n_layers - i - 1;
            let layer = Array1::<f64>::zeros(usize::pow(2, i));
            layers[r as usize] = layer;
        }

        Tree {
            size,
            total_size,
            raw: layers,
        }
    }

    pub fn empty() -> Self {
        Tree {
            size: 0,
            total_size: 0,
            raw: vec![],
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

    fn set_validated_value(&mut self, checked_idx: usize, value: f64) {
        self.raw[0][checked_idx] = value;

        let mut sub_idx = checked_idx;
        // Each parent is rebuilt as node + sibling rather than accumulating
        // deltas, so sums never drift; the new node value is already known,
        // so only the sibling needs reading.
        let mut node = value;
        for k in 1..self.raw.len() {
            node += self.raw[k - 1][sub_idx ^ 1];
            sub_idx /= 2;
            self.raw[k][sub_idx] = node;
        }
    }

    // Slow path for query: never steps into a zero-weight subtree, so it
    // lands on a positive leaf whenever the tree has positive total.
    fn descend_positive(&self, value: f64) -> i64 {
        let mut idx = 0usize;
        let mut total = 0.0;
        for layer in self.raw.iter().rev().skip(1) {
            idx *= 2;
            let left = layer[idx];
            let right = layer[idx + 1];
            if right > 0.0 && (left == 0.0 || value - total >= left) {
                total += left;
                idx += 1;
            }
        }
        idx as i64
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
        Ok(self.raw[0][self.checked_index(idx)?])
    }

    pub fn get_values(&self, idxs: &[i64]) -> Result<Vec<f64>, TreeError> {
        let idxs: Vec<usize> = idxs
            .iter()
            .map(|idx| self.checked_index(*idx))
            .collect::<Result<_, _>>()?;

        Ok(self.raw[0].select(Axis(0), &idxs).to_vec())
    }

    pub fn total(&self) -> f64 {
        *self.raw.last().expect("").get(0).expect("")
    }

    // `out` is the descent accumulator, so it is reset to the root before use.
    pub fn query(&self, values: &[f64], out: &mut [i64]) {
        assert_eq!(values.len(), out.len());
        let n = values.len();

        out.fill(0);
        let mut totals = vec![0.0; n];

        for layer in self.raw.iter().rev().skip(1) {
            for j in 0..n {
                out[j] *= 2;
                let left = layer[out[j] as usize];
                let m = values[j] - totals[j] >= left;
                totals[j] += if m { left } else { 0. };
                out[j] += if m { 1 } else { 0 };
            }
        }

        for j in 0..n {
            if self.raw[0][out[j] as usize] <= 0.0 {
                out[j] = self.descend_positive(values[j]);
            }
        }

        for idx in out.iter_mut() {
            *idx = min(*idx, (self.size - 1) as i64);
        }
    }
}
