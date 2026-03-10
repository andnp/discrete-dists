use bincode::serde::{decode_from_slice, encode_to_vec};
use ndarray::{Array1, Axis};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyTuple},
};
use serde::{Deserialize, Serialize};
use std::cmp::*;
use std::iter;

#[pyclass(module = "rust", subclass)]
#[derive(Serialize, Deserialize, Clone)]

pub struct SumTree {
    #[pyo3(get)]
    size: u32,
    total_size: u32,
    raw: Vec<Array1<f64>>,
}

impl SumTree {
    fn checked_index(&self, idx: i64) -> PyResult<usize> {
        if idx < 0 || idx >= self.size as i64 {
            return Err(PyIndexError::new_err(format!(
                "index {idx} out of bounds for SumTree of size {}",
                self.size,
            )));
        }

        Ok(idx as usize)
    }

    fn validate_weight(idx: i64, value: f64) -> PyResult<()> {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "value at index {idx} must be finite, got {value}",
            )));
        }

        if value < 0.0 {
            return Err(PyValueError::new_err(format!(
                "value at index {idx} must be nonnegative, got {value}",
            )));
        }

        Ok(())
    }

    fn set_validated_value(&mut self, checked_idx: usize, value: f64) {
        let mut sub_idx = checked_idx;
        let old = self.raw[0][sub_idx];

        self.raw.iter_mut().for_each(|level| {
            level[sub_idx] += value - old;
            sub_idx /= 2;
        });
    }
}

#[pymethods]
impl SumTree {
    #[new]
    #[pyo3(signature = (*args))]
    fn new<'py>(args: Bound<'py, PyTuple>) -> PyResult<Self> {
        match args.len() {
            0 => Ok(SumTree {
                size: 0,
                total_size: 0,
                raw: vec![],
            }),

            1 => {
                let size = args.get_item(0)?.extract::<u32>()?;

                let total_size = u32::next_power_of_two(size);
                let n_layers = u32::ilog2(total_size) + 1;

                let dummy = Array1::<f64>::zeros(1);
                let mut layers = vec![dummy; n_layers as usize];

                for i in (0..n_layers).rev() {
                    let r = n_layers - i - 1;
                    let layer = Array1::<f64>::zeros(usize::pow(2, i));
                    layers[r as usize] = layer;
                }

                Ok(SumTree {
                    size,
                    total_size,
                    raw: layers,
                })
            }

            _ => Err(PyValueError::new_err(
                "SumTree expects at most one positional argument: size",
            )),
        }
    }

    pub fn update(
        &mut self,
        idxs: PyReadonlyArray1<i64>,
        values: PyReadonlyArray1<f64>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let idxs = idxs.as_array();
        let values = values.as_array();

        if idxs.len() != values.len() {
            return Err(PyValueError::new_err(format!(
                "idxs and values must have the same length, got {} and {}",
                idxs.len(),
                values.len(),
            )));
        }

        let validated = iter::zip(idxs, values)
            .map(|(idx, value)| {
                let checked_idx = self.checked_index(*idx)?;
                Self::validate_weight(*idx, *value)?;
                Ok((checked_idx, *value))
            })
            .collect::<PyResult<Vec<_>>>()?;

        py.allow_threads(|| {
            for (checked_idx, value) in validated {
                self.set_validated_value(checked_idx, value);
            }
        });

        Ok(())
    }

    pub fn update_single(&mut self, idx: i64, value: f64) -> PyResult<()> {
        Self::validate_weight(idx, value)?;
        let checked_idx = self.checked_index(idx)?;
        self.set_validated_value(checked_idx, value);

        Ok(())
    }

    pub fn get_value(&self, idx: i64) -> PyResult<f64> {
        Ok(self.raw[0][self.checked_index(idx)?])
    }

    pub fn get_values<'py>(
        &self,
        idxs: PyReadonlyArray1<i64>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let idxs: Vec<usize> = idxs
            .as_array()
            .iter()
            .map(|idx| self.checked_index(*idx))
            .collect::<PyResult<Vec<_>>>()?;

        let values = py.allow_threads(|| self.raw[0].select(Axis(0), &idxs).to_vec());

        Ok(values.to_pyarray(py))
    }

    pub fn total(&self) -> f64 {
        *self.raw.last().expect("").get(0).expect("")
    }

    pub fn query<'py>(
        &self,
        v: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let n = v
            .len()
            .map_err(|_| PyValueError::new_err("failed to get query array length"))?;

        let v = v.as_array();
        let mut totals = Array1::<f64>::zeros(n);
        let mut idxs = Array1::<i64>::zeros(n);

        for layer in self.raw.iter().rev() {
            for j in 0..n {
                idxs[j] = idxs[j] * 2;
                let left = layer[idxs[j] as usize];

                let m = left < (v[j] - totals[j]);
                totals[j] += if m { left } else { 0. };
                idxs[j] += if m { 1 } else { 0 };
            }
        }

        idxs = idxs.map(|i| min(*i, (self.size - 1) as i64));
        Ok(idxs.to_vec().to_pyarray(py))
    }

    // enable pickling this data type
    pub fn __setstate__<'py>(&mut self, state: Bound<'py, PyBytes>) -> PyResult<()> {
        *self = decode_from_slice(state.as_bytes(), bincode::config::standard())
            .unwrap()
            .0;
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            &encode_to_vec(&self, bincode::config::standard()).unwrap(),
        ))
    }
}
