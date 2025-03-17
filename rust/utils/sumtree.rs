use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use ndarray::{Array1, Axis};
use std::iter;
use std::cmp::*;
use pyo3::{prelude::*, types::{PyBytes, PyTuple}};
use bincode::serde::{decode_from_slice, encode_to_vec};
use serde::{Deserialize, Serialize};

#[pyclass(module = "rust", subclass)]
#[derive(Serialize, Deserialize, Clone)]

pub struct SumTree {
    #[pyo3(get)]
    size: u32,
    total_size: u32,
    raw: Vec<Array1<f64>>,
}

#[pymethods]
impl SumTree {
    #[new]
    #[pyo3(signature = (*args))]
    fn new<'py>(args: Bound<'py, PyTuple>) -> Self {
        match args.len() {
            0 => SumTree {
                size: 0,
                total_size: 0,
                raw: vec![],
            },

            1 => {
                let size = args
                    .get_item(0).unwrap()
                    .extract::<u32>().unwrap();

                let total_size = u32::next_power_of_two(size);
                let n_layers = u32::ilog2(total_size) + 1;

                let dummy = Array1::<f64>::zeros(1);
                let mut layers = vec![dummy; n_layers as usize];

                for i in (0..n_layers).rev() {
                    let r = n_layers - i - 1;
                    let layer = Array1::<f64>::zeros(usize::pow(2, i));
                    layers[r as usize] = layer;
                }

                SumTree {
                    size,
                    total_size,
                    raw: layers,
                }
            },

            _ => unreachable!(),
        }
    }

    pub fn update(
        &mut self,
        idxs: PyReadonlyArray1<i64>,
        values: PyReadonlyArray1<f64>,
    ) {
        iter::zip(idxs.as_array(), values.as_array())
            .for_each(|(idx, v)| { self.update_single(*idx, *v) });
    }

    pub fn update_single(
        &mut self,
        idx: i64,
        value: f64,
    ) {
        if idx >= self.size as i64 {
            panic!("Tried to update index outside of tree: <{idx}>");
        }

        let mut sub_idx = idx as usize;
        let old = self.raw[0][sub_idx];

        self.raw.iter_mut()
            .for_each(|level| {
                level[sub_idx] += value - old;
                sub_idx = sub_idx / 2;
            });
    }

    pub fn get_value(&mut self, idx: i64) -> f64 {
        self.raw[0][idx as usize]
    }

    pub fn get_values<'py>(
        &mut self,
        idxs: PyReadonlyArray1<i64>,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        let idxs = idxs.as_array().map(|a| { *a as usize });
        let arr = self.raw[0]
            .select(Axis(0), &idxs.to_vec());

            // arr.to_pyarray(py)
            arr.to_vec().to_pyarray(py)
    }

    pub fn total(
        &mut self,
    ) -> f64 {
        *self.raw
            .last()
            .expect("")
            .get(0)
            .expect("")
    }

    pub fn query<'py>(
        &mut self,
        v: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<i64>> {
        let n = v.len().expect("Failed to get array length");

        let v = v.as_array();
        let mut totals = Array1::<f64>::zeros(n);
        let mut idxs = Array1::<i64>::zeros(n);

        self.raw.iter()
            .rev()
            .for_each(|layer| {
                for j in 0..n {
                    idxs[j] = idxs[j] * 2;
                    let left = *layer
                        .get(idxs[j] as usize)
                        .expect("");

                    let m = left < (v[j] - totals[j]);
                    totals[j] += if m { left } else { 0. };
                    idxs[j] += if m { 1 } else { 0 };
                }
            });

        idxs = idxs.map(|i| { min(*i, (self.size - 1) as i64) });
        idxs.to_vec().to_pyarray(py)
    }

    // enable pickling this data type
    pub fn __setstate__<'py>(&mut self, state: Bound<'py, PyBytes>) -> PyResult<()> {
        *self = decode_from_slice(state.as_bytes(), bincode::config::standard()).unwrap().0;
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &encode_to_vec(&self, bincode::config::standard()).unwrap()))
    }
}
