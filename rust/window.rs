use crate::rng::Rng;
use crate::utils::sumtree::{contiguous, to_py_err};
use bincode::serde::{decode_from_slice, encode_to_vec};
use discrete_dists_core::{Distribution, Window, WindowMixture as CoreWindowMixture};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::{prelude::*, types::PyBytes};

type WindowSpec = (f64, f64, Option<f64>, f64);

#[pyclass(module = "discrete_dists.rust")]
pub struct WindowMixture {
    inner: CoreWindowMixture,
}

#[pymethods]
impl WindowMixture {
    #[new]
    fn new(size: u32, windows: Vec<WindowSpec>, cursor: f64) -> PyResult<Self> {
        let windows = windows
            .into_iter()
            .map(|(min_age, max_age, half_life, p)| Window {
                min_age,
                max_age,
                half_life,
                p,
            })
            .collect();
        let inner = CoreWindowMixture::new(size, windows, cursor).map_err(to_py_err)?;

        Ok(WindowMixture { inner })
    }

    #[getter]
    fn cursor(&self) -> f64 {
        self.inner.cursor()
    }

    #[getter]
    fn counts(&self) -> Vec<u64> {
        self.inner.counts().to_vec()
    }

    #[getter]
    fn is_defunct(&self) -> bool {
        self.inner.is_defunct()
    }

    pub fn insert(
        &mut self,
        elements: PyReadonlyArray1<i64>,
        stamps: PyReadonlyArray1<f64>,
        weights: PyReadonlyArray1<f64>,
        py: Python<'_>,
    ) -> PyResult<()> {
        let elements = contiguous(&elements);
        let stamps = contiguous(&stamps);
        let weights = contiguous(&weights);

        py.detach(|| self.inner.insert(&elements, &stamps, &weights))
            .map_err(to_py_err)
    }

    pub fn remove(&mut self, elements: PyReadonlyArray1<i64>, py: Python<'_>) -> PyResult<()> {
        let elements = contiguous(&elements);

        py.detach(|| self.inner.remove(&elements))
            .map_err(to_py_err)
    }

    pub fn advance(&mut self, to: f64, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.advance(to)).map_err(to_py_err)
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
    fn __getnewargs__(&self) -> (u32, Vec<WindowSpec>, f64) {
        (1, vec![(0.0, f64::INFINITY, None, 1.0)], 0.0)
    }

    pub fn __setstate__<'py>(&mut self, state: Bound<'py, PyBytes>) -> PyResult<()> {
        let (inner, _) = decode_from_slice(state.as_bytes(), bincode::config::standard())
            .map_err(|e| PyValueError::new_err(format!("invalid WindowMixture state: {e}")))?;
        self.inner = inner;
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = encode_to_vec(&self.inner, bincode::config::standard()).map_err(|e| {
            PyValueError::new_err(format!("failed to serialize WindowMixture: {e}"))
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
