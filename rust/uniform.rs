use crate::rng::Rng;
use crate::utils::sumtree::{contiguous, to_py_err};
use discrete_dists_core::{Distribution, Uniform};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn uniform(lo: i64, hi: i64) -> PyResult<Uniform> {
    if hi < lo {
        return Err(PyValueError::new_err(format!(
            "support [{lo}, {hi}) is reversed"
        )));
    }

    Ok(Uniform::new(lo, hi))
}

#[pyfunction]
pub fn uniform_sample<'py>(
    mut rng: PyRefMut<'_, Rng>,
    lo: i64,
    hi: i64,
    n: usize,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let dist = uniform(lo, hi)?;
    let core = &mut rng.rng;
    let mut out = vec![0i64; n];
    py.detach(|| dist.sample(core, &mut out))
        .map_err(to_py_err)?;

    Ok(out.to_pyarray(py))
}

#[pyfunction]
pub fn uniform_stratified_sample<'py>(
    mut rng: PyRefMut<'_, Rng>,
    lo: i64,
    hi: i64,
    n: usize,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let dist = uniform(lo, hi)?;
    let core = &mut rng.rng;
    let mut out = vec![0i64; n];
    py.detach(|| dist.stratified_sample(core, &mut out))
        .map_err(to_py_err)?;

    Ok(out.to_pyarray(py))
}

#[pyfunction]
pub fn uniform_sample_without_replacement<'py>(
    mut rng: PyRefMut<'_, Rng>,
    lo: i64,
    hi: i64,
    n: usize,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let dist = uniform(lo, hi)?;
    let core = &mut rng.rng;
    let mut out = Vec::with_capacity(n);
    py.detach(|| dist.sample_without_replacement(core, n, &mut out))
        .map_err(to_py_err)?;

    Ok(out.to_pyarray(py))
}

#[pyfunction]
pub fn uniform_probs<'py>(
    lo: i64,
    hi: i64,
    elements: PyReadonlyArray1<i64>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let dist = uniform(lo, hi)?;
    let elements = contiguous(&elements);
    let mut out = vec![0.0; elements.len()];
    dist.probs(&elements, &mut out);

    Ok(out.to_pyarray(py))
}
