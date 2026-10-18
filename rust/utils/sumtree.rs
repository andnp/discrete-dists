use crate::rng::Rng;
use bincode::serde::{decode_from_slice, encode_to_vec};
use discrete_dists_core::{Distribution, Proportional, Tree, TreeError};
use numpy::{Element, PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyTuple},
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

#[pyclass(module = "rust", subclass)]
#[derive(Serialize, Deserialize, Clone)]
#[serde(transparent)]
pub struct SumTree {
    pub(crate) tree: Tree,
}

#[cold]
#[inline(never)]
pub(crate) fn to_py_err(e: TreeError) -> PyErr {
    match e {
        TreeError::IndexOutOfBounds { .. } | TreeError::OutOfSupport { .. } => {
            PyIndexError::new_err(e.to_string())
        }
        _ => PyValueError::new_err(e.to_string()),
    }
}

pub(crate) fn contiguous<'a, 'py, T: Element + Clone>(
    arr: &'a PyReadonlyArray1<'py, T>,
) -> Cow<'a, [T]> {
    match arr.as_slice() {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(arr.as_array().to_vec()),
    }
}

#[pymethods]
impl SumTree {
    #[new]
    #[pyo3(signature = (*args))]
    fn new<'py>(args: Bound<'py, PyTuple>) -> PyResult<Self> {
        match args.len() {
            0 => Ok(SumTree {
                tree: Tree::empty(),
            }),

            1 => {
                let size = args.get_item(0)?.extract::<u32>()?;
                Ok(SumTree {
                    tree: Tree::new(size),
                })
            }

            _ => Err(PyValueError::new_err(
                "SumTree expects at most one positional argument: size",
            )),
        }
    }

    #[getter]
    fn size(&self) -> u32 {
        self.tree.size()
    }

    pub fn update(
        &mut self,
        idxs: PyReadonlyArray1<i64>,
        values: PyReadonlyArray1<f64>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let idxs = contiguous(&idxs);
        let values = contiguous(&values);

        py.detach(|| self.tree.update(&idxs, &values))
            .map_err(to_py_err)
    }

    pub fn update_single(&mut self, idx: i64, value: f64) -> PyResult<()> {
        self.tree.update_single(idx, value).map_err(to_py_err)
    }

    pub fn scale(&mut self, factor: f64) -> PyResult<()> {
        self.tree.scale(factor).map_err(to_py_err)
    }

    pub fn get_value(&self, idx: i64) -> PyResult<f64> {
        self.tree.get_value(idx).map_err(to_py_err)
    }

    pub fn get_values<'py>(
        &self,
        idxs: PyReadonlyArray1<i64>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let idxs = contiguous(&idxs);

        let values = py
            .detach(|| self.tree.get_values(&idxs))
            .map_err(to_py_err)?;

        Ok(values.to_pyarray(py))
    }

    #[pyo3(signature = (rng, n, offset = 0))]
    pub fn sample<'py>(
        &self,
        mut rng: PyRefMut<'_, Rng>,
        n: usize,
        offset: i64,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let core = &mut rng.rng;
        let mut out = vec![0i64; n];
        py.detach(|| Proportional::from_tree(offset, &self.tree).sample(core, &mut out))
            .map_err(to_py_err)?;

        Ok(out.to_pyarray(py))
    }

    #[pyo3(signature = (rng, n, offset = 0))]
    pub fn stratified_sample<'py>(
        &self,
        mut rng: PyRefMut<'_, Rng>,
        n: usize,
        offset: i64,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let core = &mut rng.rng;
        let mut out = vec![0i64; n];
        py.detach(|| Proportional::from_tree(offset, &self.tree).stratified_sample(core, &mut out))
            .map_err(to_py_err)?;

        Ok(out.to_pyarray(py))
    }

    #[pyo3(signature = (rng, n, offset = 0))]
    pub fn sample_without_replacement<'py>(
        &mut self,
        mut rng: PyRefMut<'_, Rng>,
        n: usize,
        offset: i64,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let core = &mut rng.rng;
        let mut out = Vec::with_capacity(n);
        py.detach(|| {
            Proportional::from_tree(offset, &mut self.tree)
                .sample_without_replacement(core, n, &mut out)
        })
        .map_err(to_py_err)?;

        Ok(out.to_pyarray(py))
    }

    pub fn total(&self) -> f64 {
        self.tree.total()
    }

    pub fn query<'py>(
        &self,
        v: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let n = v
            .len()
            .map_err(|_| PyValueError::new_err("failed to get query array length"))?;

        let values = contiguous(&v);
        let mut out = vec![0i64; n];
        self.tree.query(&values, &mut out);

        Ok(out.to_pyarray(py))
    }

    // enable pickling this data type
    pub fn __setstate__<'py>(&mut self, state: Bound<'py, PyBytes>) -> PyResult<()> {
        let (tree, _) = decode_from_slice(state.as_bytes(), bincode::config::standard())
            .map_err(|e| PyValueError::new_err(format!("invalid SumTree state: {e}")))?;
        *self = tree;
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = encode_to_vec(self, bincode::config::standard())
            .map_err(|e| PyValueError::new_err(format!("failed to serialize SumTree: {e}")))?;
        Ok(PyBytes::new(py, &bytes))
    }
}

#[pyfunction]
pub fn proportional_update(
    lo: i64,
    mut tree: PyRefMut<'_, SumTree>,
    elements: PyReadonlyArray1<i64>,
    values: PyReadonlyArray1<f64>,
) -> PyResult<()> {
    let elements = contiguous(&elements);
    let values = contiguous(&values);
    Proportional::from_tree(lo, &mut tree.tree)
        .update(&elements, &values)
        .map_err(to_py_err)
}

#[pyfunction]
pub fn proportional_probs<'py>(
    lo: i64,
    tree: PyRef<'_, SumTree>,
    elements: PyReadonlyArray1<i64>,
    py: Python<'py>,
) -> Bound<'py, PyArray1<f64>> {
    let elements = contiguous(&elements);
    let mut out = vec![0.0; elements.len()];
    Proportional::from_tree(lo, &tree.tree).probs(&elements, &mut out);

    out.to_pyarray(py)
}
