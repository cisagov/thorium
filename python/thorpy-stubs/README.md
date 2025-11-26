# thorpy-stubs

Automatically generates a `.pyi` stubs file for [thorpy](../thorpy/README.md) using
`PyO3`'s experimental introspection features.

## How to use

`thorpy-stubs` requires `thorpy` to be built for development first to introspect and
generate function stubs. Follow the instructions to build `thorpy` in
the
[thorpy docs](../thorpy/README.md#building). You need `uv` and `maturin` installed and a
virtual environment set in the `thorpy` directory. Install `thorpy` in the virtual environment
by running:

```Bash
maturin develop
```

Then run `thorpy-stubs` to locate the library built in the previous command
and generate a stubs file from it:

```Bash
cargo run --package thorpy-stubs
```

This will create a stubs file called `thorpy.pyi` in the `thorpy` directory. Subsequent
builds using `maturin` will detect the stubs file and include it in built wheels.

You'll need to generate a new stubs file if you modify the `thorpy` code. Be sure to build
the latest `thorpy` client using `maturin develop` before generating the stubs file.

## Limitations

Automatic stub generation through type introspection is still experiemental. The following
are the major limitations in the stubs we generate:

- Enum variants are missing
- Class properties generated with `#[pyo3(get)]` and `#[pyo3(set)]` don't have typing
info
- Correct typing in function signatures oftentimes requires explicitly writing out
the function signature using `#[pyo3(signature = ... )]` which may lead to mismatches
as functions are updated

## Resources

- [PyO3 type stub generation docs](https://pyo3.rs/v0.27.2/type-stub.html)
- [PyO3 typing hints docs](https://pyo3.rs/v0.27.2/python-typing-hints.html)
- [maturin docs](https://www.maturin.rs/index.html)
