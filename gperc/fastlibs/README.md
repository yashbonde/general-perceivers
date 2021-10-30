# Fastlib

Sometimes you need the power and functionality that entirity of python cannot do it, plus it's cool so I am adding a few methods in Rust. I am using [`pyo3`](https://github.com/PyO3/pyo3) for this example, these are the steps to be followed:

1. In `cargo.toml` whatever `name = "gperc_fast"` in `[lib]` you define will be the package name and it has to be same as `#[pymodule]` in `src/lib.rs`.
2. Install `maturin` and to build use `maturin develop` which will put binaries in the python3 `/bin` by default. Need to see how to control that behaviour.
