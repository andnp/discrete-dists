use pyo3::prelude::*;

mod rng;
mod uniform;
mod utils;

/// A Python module implemented in Rust.
#[pymodule(gil_used = false)]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<rng::Rng>()?;
    m.add_class::<utils::sumtree::SumTree>()?;
    m.add_function(wrap_pyfunction!(uniform::uniform_sample, m)?)?;
    m.add_function(wrap_pyfunction!(uniform::uniform_stratified_sample, m)?)?;
    m.add_function(wrap_pyfunction!(
        uniform::uniform_sample_without_replacement,
        m
    )?)?;
    Ok(())
}
