use crate::rng::Rng;
use crate::uniform::uniform;
use crate::uniform_set::UniformSet;
use crate::utils::sumtree::{to_py_err, SumTree};
use discrete_dists_core::{
    mixture, Distribution, Proportional, Tree, TreeError, Uniform, UniformSet as CoreUniformSet,
};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

enum Source<'py> {
    Uniform(Uniform),
    Proportional(i64, PyRef<'py, SumTree>),
    UniformSet(PyRef<'py, UniformSet>),
}

enum Part<'a> {
    Uniform(Uniform),
    Proportional(Proportional<&'a Tree>),
    UniformSet(&'a CoreUniformSet),
}

impl Part<'_> {
    fn as_dist(&self) -> &dyn Distribution {
        match self {
            Part::Uniform(u) => u,
            Part::Proportional(p) => p,
            Part::UniformSet(s) => *s,
        }
    }
}

fn parse_sources<'py>(parts: &Bound<'py, PyList>) -> PyResult<Vec<(Source<'py>, f64)>> {
    parts
        .iter()
        .map(|item| {
            let (kind, payload, weight): (String, Bound<'py, PyAny>, f64) = item.extract()?;
            let source = match kind.as_str() {
                "uniform" => {
                    let (lo, hi): (i64, i64) = payload.extract()?;
                    Source::Uniform(uniform(lo, hi)?)
                }
                "proportional" => {
                    let (lo, tree): (i64, PyRef<'py, SumTree>) = payload.extract()?;
                    Source::Proportional(lo, tree)
                }
                "uniform_set" => Source::UniformSet(payload.extract()?),
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown mixture part kind {other:?}"
                    )))
                }
            };

            Ok((source, weight))
        })
        .collect()
}

// The tree borrows must outlive every part, so the sources are parsed first and the parts view them.
fn with_parts<R>(
    parts: &Bound<'_, PyList>,
    f: impl FnOnce(&[(&dyn Distribution, f64)]) -> Result<R, TreeError>,
) -> PyResult<R> {
    let sources = parse_sources(parts)?;
    let views: Vec<(Part<'_>, f64)> = sources
        .iter()
        .map(|(source, weight)| {
            let part = match source {
                Source::Uniform(u) => Part::Uniform(u.clone()),
                Source::Proportional(lo, tree) => {
                    Part::Proportional(Proportional::from_tree(*lo, &tree.tree))
                }
                Source::UniformSet(set) => Part::UniformSet(&set.inner),
            };

            (part, *weight)
        })
        .collect();
    let dists: Vec<(&dyn Distribution, f64)> = views
        .iter()
        .map(|(part, weight)| (part.as_dist(), *weight))
        .collect();

    f(&dists).map_err(to_py_err)
}

#[pyfunction]
pub fn mixture_sample<'py>(
    mut rng: PyRefMut<'_, Rng>,
    parts: Bound<'py, PyList>,
    n: usize,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let core = &mut rng.rng;
    let mut out = vec![0i64; n];
    with_parts(&parts, |dists| mixture::sample(dists, core, &mut out))?;

    Ok(out.to_pyarray(py))
}

#[pyfunction]
pub fn mixture_stratified_sample<'py>(
    mut rng: PyRefMut<'_, Rng>,
    parts: Bound<'py, PyList>,
    n: usize,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let core = &mut rng.rng;
    let mut out = vec![0i64; n];
    with_parts(&parts, |dists| {
        mixture::stratified_sample(dists, core, &mut out)
    })?;

    Ok(out.to_pyarray(py))
}

#[pyfunction]
pub fn mixture_sample_without_replacement<'py>(
    mut rng: PyRefMut<'_, Rng>,
    parts: Bound<'py, PyList>,
    n: usize,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let core = &mut rng.rng;
    let out = with_parts(&parts, |dists| {
        mixture::sample_without_replacement(dists, core, n)
    })?;

    Ok(out.to_pyarray(py))
}

#[pyfunction]
pub fn mixture_probs<'py>(
    parts: Bound<'py, PyList>,
    elements: PyReadonlyArray1<i64>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let elements = elements.as_array().to_vec();
    let mut out = vec![0.0; elements.len()];
    with_parts(&parts, |dists| {
        mixture::probs(dists, &elements, &mut out);
        Ok(())
    })?;

    Ok(out.to_pyarray(py))
}
