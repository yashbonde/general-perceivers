use pyo3::prelude::*;
use std::io::prelude::*;

use std::fs;
use std::os::unix::prelude::FileExt;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

// take an array of i64 and return sum
#[pyfunction]
fn sum_array(a: Vec<i64>) -> PyResult<i64> {
    Ok(a.iter().sum())
}

// struct Sample {
//     filepath: String,
//     start: u16,
//     end: u16,
// }

#[pyfunction]
fn buffered_read(filepath: String, start: u64, end: u64) -> PyResult<Vec<i64>> {
    let mut f = fs::File::open(filepath)?;
    let mut buffer = [0; 128];

    // read up to 10 bytes
    f.read_at(&mut buffer, start)?;
    Ok(buffer.to_vec())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn gperc_fast(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(sum_array, m)?)?;
    m.add_function(wrap_pyfunction!(buffered_read, m)?)?;

    Ok(())
}
