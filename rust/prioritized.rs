use crate::rng::Rng;
use crate::utils::sumtree::{contiguous, to_py_err};
use bincode::serde::{decode_from_slice, encode_to_vec};
use discrete_dists_core::{Distribution, Prioritized as CorePrioritized};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::{prelude::*, types::PyBytes};

type SampleWithWeights<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>);

#[pyclass(module = "discrete_dists.rust")]
pub struct Prioritized {
    inner: CorePrioritized,
}

#[pymethods]
impl Prioritized {
    #[new]
    fn new(size: u32, alpha: f64) -> PyResult<Self> {
        let inner = CorePrioritized::new(size, alpha).map_err(to_py_err)?;

        Ok(Prioritized { inner })
    }

    #[getter]
    fn size(&self) -> u32 {
        self.inner.size()
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }

    #[getter]
    fn max_priority(&self) -> Option<f64> {
        self.inner.max_priority()
    }

    #[getter]
    fn is_defunct(&self) -> bool {
        self.inner.is_defunct()
    }

    pub fn update(
        &mut self,
        elements: PyReadonlyArray1<i64>,
        priorities: PyReadonlyArray1<f64>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let elements = contiguous(&elements);
        let priorities = contiguous(&priorities);

        py.detach(|| self.inner.update(&elements, &priorities))
            .map_err(to_py_err)
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

    pub fn importance_weights<'py>(
        &self,
        elements: PyReadonlyArray1<i64>,
        beta: f64,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let elements = contiguous(&elements);
        let mut out = vec![0.0; elements.len()];
        py.detach(|| self.inner.importance_weights(&elements, beta, &mut out))
            .map_err(to_py_err)?;

        Ok(out.to_pyarray(py))
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
        &mut self,
        mut rng: PyRefMut<'_, Rng>,
        n: usize,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let core = &mut rng.rng;
        let mut out = Vec::with_capacity(n);
        py.detach(|| self.inner.sample_without_replacement(core, n, &mut out))
            .map_err(to_py_err)?;

        Ok(out.to_pyarray(py))
    }

    #[pyo3(signature = (rng, n, beta, stratified = false))]
    pub fn sample_with_weights<'py>(
        &self,
        mut rng: PyRefMut<'_, Rng>,
        n: usize,
        beta: f64,
        stratified: bool,
        py: Python<'py>,
    ) -> PyResult<SampleWithWeights<'py>> {
        let core = &mut rng.rng;
        let mut out = vec![0i64; n];
        let mut weights = vec![0.0; n];
        py.detach(|| {
            self.inner
                .sample_with_weights(core, beta, stratified, &mut out, &mut weights)
        })
        .map_err(to_py_err)?;

        Ok((out.to_pyarray(py), weights.to_pyarray(py)))
    }

    // Unpickling calls `__new__` with these args before `__setstate__` restores the state.
    fn __getnewargs__(&self) -> (u32, f64) {
        (1, 1.0)
    }

    pub fn __setstate__<'py>(&mut self, state: Bound<'py, PyBytes>) -> PyResult<()> {
        let (inner, _) = decode_from_slice(state.as_bytes(), bincode::config::standard())
            .map_err(|e| PyValueError::new_err(format!("invalid Prioritized state: {e}")))?;
        self.inner = inner;
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = encode_to_vec(&self.inner, bincode::config::standard())
            .map_err(|e| PyValueError::new_err(format!("failed to serialize Prioritized: {e}")))?;
        Ok(PyBytes::new(py, &bytes))
    }
}
