use pyo3::prelude::*;

mod utils;

/// A Python module implemented in Rust.
#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<utils::sumtree::SumTree>()?;
    Ok(())
}
