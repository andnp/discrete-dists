use pyo3::prelude::*;

mod alias;
mod mixture;
mod prioritized;
mod reservoir;
mod rng;
mod uniform;
mod uniform_set;
mod utils;
mod window;

/// A Python module implemented in Rust.
#[pymodule(gil_used = false)]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<rng::Rng>()?;
    m.add_class::<alias::Alias>()?;
    m.add_class::<prioritized::Prioritized>()?;
    m.add_class::<reservoir::WeightedReservoir>()?;
    m.add_class::<utils::sumtree::SumTree>()?;
    m.add_class::<uniform_set::UniformSet>()?;
    m.add_class::<window::WindowMixture>()?;
    m.add_function(wrap_pyfunction!(mixture::mixture_sample, m)?)?;
    m.add_function(wrap_pyfunction!(mixture::mixture_stratified_sample, m)?)?;
    m.add_function(wrap_pyfunction!(
        mixture::mixture_sample_without_replacement,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(mixture::mixture_probs, m)?)?;
    m.add_function(wrap_pyfunction!(utils::sumtree::proportional_update, m)?)?;
    m.add_function(wrap_pyfunction!(utils::sumtree::proportional_probs, m)?)?;
    m.add_function(wrap_pyfunction!(uniform::uniform_probs, m)?)?;
    m.add_function(wrap_pyfunction!(uniform::uniform_sample, m)?)?;
    m.add_function(wrap_pyfunction!(uniform::uniform_stratified_sample, m)?)?;
    m.add_function(wrap_pyfunction!(
        uniform::uniform_sample_without_replacement,
        m
    )?)?;
    Ok(())
}
