use pyo3::prelude::*;

mod rng;
mod utils;

/// A Python module implemented in Rust.
#[pymodule(gil_used = false)]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<rng::Rng>()?;
    m.add_class::<utils::sumtree::SumTree>()?;
    Ok(())
}
