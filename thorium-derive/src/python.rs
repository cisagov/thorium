//! Helper proc-macros to use in together with ``PyO3`` to generate Python
//! methods and classes

mod pyclass;
mod pyenum;

pub use pyclass::pyclass;
pub use pyenum::pyenum;

use quote::quote;
use syn::{Type, TypePath};

/// Takes a [`Type`] and provides a function to convert
/// it or its inner values (i.e. `Vec`, `Option`, `Result`)
fn type_into(ty: &Type) -> proc_macro2::TokenStream {
    let (ty_ident, _generics) = match ty {
        Type::Path(TypePath { qself: None, path }) => {
            let seg = path
                .segments
                .last()
                .unwrap_or_else(|| panic!("found empty path in type: '{ty:?}'"));
            (&seg.ident, &seg.arguments)
        }
        // anything else (reference, array, tuple, …) fall back to just `.into()`
        _ => return quote! { .into() },
    };
    match ty_ident.to_string().as_str() {
        "Vec" => {
            quote! {
                .into_iter()
                    .map(std::convert::Into::into)
                    .collect()
            }
        }
        "Option" | "Result" => {
            quote! {
                .map(std::convert::Into::into)
            }
        }
        _ => quote! { .into() },
    }
}
