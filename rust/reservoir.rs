use crate::rng::Rng;
use crate::utils::sumtree::{contiguous, to_py_err};
use bincode::serde::{decode_from_slice, encode_to_vec};
use discrete_dists_core::WeightedReservoir as CoreWeightedReservoir;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::{prelude::*, types::PyBytes};

type Offered<'py> = (Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<i64>>);

#[pyclass(module = "discrete_dists.rust")]
pub struct WeightedReservoir {
    inner: CoreWeightedReservoir,
}

#[pymethods]
impl WeightedReservoir {
    #[new]
    fn new(capacity: u32) -> Self {
        WeightedReservoir {
            inner: CoreWeightedReservoir::new(capacity),
        }
    }

    #[getter]
    fn capacity(&self) -> u32 {
        self.inner.capacity()
    }

    #[getter]
    fn elements<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.elements().to_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }

    pub fn offer<'py>(
        &mut self,
        mut rng: PyRefMut<'_, Rng>,
        elements: PyReadonlyArray1<i64>,
        weights: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> PyResult<Offered<'py>> {
        let elements = contiguous(&elements);
        let weights = contiguous(&weights);
        let mut admitted = vec![false; elements.len()];
        let mut evicted = Vec::new();

        let core = &mut rng.rng;
        py.detach(|| {
            self.inner
                .offer_batch(core, &elements, &weights, &mut admitted, &mut evicted)
        })
        .map_err(to_py_err)?;

        Ok((admitted.to_pyarray(py), evicted.to_pyarray(py)))
    }

    // Unpickling calls `__new__` with these args before `__setstate__` restores the state.
    fn __getnewargs__(&self) -> (u32,) {
        (1,)
    }

    pub fn __setstate__<'py>(&mut self, state: Bound<'py, PyBytes>) -> PyResult<()> {
        let (inner, _) = decode_from_slice(state.as_bytes(), bincode::config::standard())
            .map_err(|e| PyValueError::new_err(format!("invalid WeightedReservoir state: {e}")))?;
        self.inner = inner;
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = encode_to_vec(&self.inner, bincode::config::standard()).map_err(|e| {
            PyValueError::new_err(format!("failed to serialize WeightedReservoir: {e}"))
        })?;
        Ok(PyBytes::new(py, &bytes))
    }
}
