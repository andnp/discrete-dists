use crate::rng::Rng;
use crate::utils::sumtree::{contiguous, to_py_err};
use bincode::serde::{decode_from_slice, encode_to_vec};
use discrete_dists_core::{Alias as CoreAlias, Distribution};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::{prelude::*, types::PyBytes};

#[pyclass(module = "discrete_dists.rust")]
pub struct Alias {
    pub(crate) inner: CoreAlias,
}

#[pymethods]
impl Alias {
    #[new]
    #[pyo3(signature = (weights, lo = 0))]
    fn new(weights: PyReadonlyArray1<f64>, lo: i64) -> PyResult<Self> {
        let weights = contiguous(&weights);
        let inner = CoreAlias::new(lo, &weights).map_err(to_py_err)?;

        Ok(Alias { inner })
    }

    #[getter]
    fn support(&self) -> (i64, i64) {
        self.inner.support()
    }

    #[getter]
    fn is_defunct(&self) -> bool {
        self.inner.is_defunct()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn probs<'py>(
        &self,
        elements: PyReadonlyArray1<i64>,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        let elements = contiguous(&elements);
        let mut out = vec![0.0; elements.len()];
        py.detach(|| self.inner.probs(&elements, &mut out));

        out.to_pyarray(py)
    }

    pub fn sample<'py>(
        &self,
        mut rng: PyRefMut<'_, Rng>,
        n: usize,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let core = &mut rng.rng;
        let mut out = vec![0i64; n];
        py.detach(|| self.inner.sample(core, &mut out))
            .map_err(to_py_err)?;

        Ok(out.to_pyarray(py))
    }

    pub fn stratified_sample<'py>(
        &self,
        mut rng: PyRefMut<'_, Rng>,
        n: usize,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let core = &mut rng.rng;
        let mut out = vec![0i64; n];
        py.detach(|| self.inner.stratified_sample(core, &mut out))
            .map_err(to_py_err)?;

        Ok(out.to_pyarray(py))
    }

    pub fn sample_without_replacement<'py>(
        &self,
        mut rng: PyRefMut<'_, Rng>,
        n: usize,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let core = &mut rng.rng;
        let out = py
            .detach(|| self.inner.sample_without_replacement(core, n))
            .map_err(to_py_err)?;

        Ok(out.to_pyarray(py))
    }

    // Unpickling calls `__new__` with these args before `__setstate__` restores the state.
    fn __getnewargs__<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray1<f64>>, i64) {
        (PyArray1::from_slice(py, &[1.0]), 0)
    }

    pub fn __setstate__<'py>(&mut self, state: Bound<'py, PyBytes>) -> PyResult<()> {
        let (inner, _) = decode_from_slice(state.as_bytes(), bincode::config::standard())
            .map_err(|e| PyValueError::new_err(format!("invalid Alias state: {e}")))?;
        self.inner = inner;
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = encode_to_vec(&self.inner, bincode::config::standard())
            .map_err(|e| PyValueError::new_err(format!("failed to serialize Alias: {e}")))?;
        Ok(PyBytes::new(py, &bytes))
    }
}
