use bincode::serde::{decode_from_slice, encode_to_vec};
use discrete_dists_core::Rng as CoreRng;
use numpy::{PyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::{prelude::*, types::PyBytes};

#[pyclass(module = "discrete_dists.rust")]
pub struct Rng {
    pub(crate) rng: CoreRng,
}

#[pymethods]
impl Rng {
    #[new]
    fn new(seed: i128) -> PyResult<Self> {
        let seed = u64::try_from(seed).map_err(|_| {
            PyValueError::new_err(format!("seed must be in [0, 2**64), got {seed}"))
        })?;

        Ok(Rng {
            rng: CoreRng::new(seed),
        })
    }

    pub fn random<'py>(&mut self, n: usize, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let out: Vec<f64> = (0..n).map(|_| self.rng.next_f64()).collect();

        out.to_pyarray(py)
    }

    // Unpickling calls `Rng.__new__` with these args before `__setstate__` restores the stream.
    fn __getnewargs__(&self) -> (u64,) {
        (0,)
    }

    pub fn __setstate__<'py>(&mut self, state: Bound<'py, PyBytes>) -> PyResult<()> {
        let (rng, _) = decode_from_slice(state.as_bytes(), bincode::config::standard())
            .map_err(|e| PyValueError::new_err(format!("invalid Rng state: {e}")))?;
        *self = Rng { rng };
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = encode_to_vec(&self.rng, bincode::config::standard())
            .map_err(|e| PyValueError::new_err(format!("failed to serialize Rng: {e}")))?;
        Ok(PyBytes::new(py, &bytes))
    }
}
