//! Helper proc-macros to use in together with ``PyO3`` to generate Python
//! methods and classes

use darling::FromMeta;
use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{Attribute, Error, ItemStruct, Visibility, parse_quote};

/// Arguments to the blocking subclient attribute
#[derive(Debug, Clone, Copy, Default, FromMeta)]
#[darling(derive_syn_parse, default)]
struct PyClassArgs {
    /// Whether the proc macro should automatically set the `#[pyo3(get)]`
    /// attribute for public fields
    get: bool,
    /// Whether the proc macro should automatically set the `#[pyo3(set)]`
    /// attribute for public fields
    set: bool,
}

impl PyClassArgs {
    fn field_attr(self) -> Option<Attribute> {
        match (self.get, self.set) {
            (true, true) => Some(parse_quote!(#[pyo3(get, set)])),
            (true, false) => Some(parse_quote!(#[pyo3(get)])),
            (false, true) => Some(parse_quote!(#[pyo3(set)])),
            (false, false) => None,
        }
    }
}

/// Set the struct to be a `pyclass`, applying `get` and `set` attributes to
/// public fields depending on the given args
pub fn pyclass(args_raw: TokenStream, input: TokenStream) -> TokenStream {
    let mut input_struct = match syn::parse::<ItemStruct>(input) {
        Ok(input_struct) => input_struct,
        Err(err) => {
            return Error::new(err.span(), "expected a struct")
                .to_compile_error()
                .into();
        }
    };

    let args: PyClassArgs = match syn::parse(args_raw) {
        Ok(args) => args,
        Err(err) => return err.to_compile_error().into(),
    };

    // append the #[pyclass] attribute to the struct
    let pyclass_attr: Attribute = parse_quote!(#[pyclass]);
    input_struct.attrs.push(pyclass_attr);

    // add pyo3 attributes to each public field depending on what we have set
    if let Some(pyo3_field_attr) = args.field_attr() {
        match &mut input_struct.fields {
            syn::Fields::Named(named) => {
                for field in &mut named.named {
                    if matches!(field.vis, Visibility::Public(_)) {
                        field.attrs.push(pyo3_field_attr.clone());
                    }
                }
            }
            syn::Fields::Unnamed(unnamed) => {
                for field in &mut unnamed.unnamed {
                    if matches!(field.vis, Visibility::Public(_)) {
                        field.attrs.push(pyo3_field_attr.clone());
                    }
                }
            }
            syn::Fields::Unit => {}
        }
    }

    quote!(#input_struct).into_token_stream().into()
}
