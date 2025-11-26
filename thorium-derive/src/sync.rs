//! proc-macros for generating a blocking sync client from an async one

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::{Attribute, Block, Error, ImplItemFn, ItemFn, Signature, Visibility};

mod sync_struct;
mod sync_trait;

pub use sync_struct::blocking_struct;
pub use sync_trait::blocking_trait;

/// Create a blocking version of the async function
///
/// The body of the async function is just blocked on the static runtime
pub fn blocking_fn(_meta: TokenStream, input: TokenStream) -> TokenStream {
    if let Ok(item_fn) = syn::parse::<ItemFn>(input.clone()) {
        // get blocking function
        let blocking_fn =
            match blocking_fn_helper(&item_fn.attrs, &item_fn.vis, &item_fn.sig, &item_fn.block) {
                Ok(blocking_fn) => blocking_fn,
                Err(err) => return err.into_compile_error().into(),
            };
        // return the regular function and the blocking function
        quote! {
            #item_fn

            #blocking_fn
        }
        .into()
    } else if let Ok(impl_item_fn) = syn::parse::<ImplItemFn>(input) {
        // get the blocking function
        let blocking_fn = match blocking_fn_helper(
            &impl_item_fn.attrs,
            &impl_item_fn.vis,
            &impl_item_fn.sig,
            &impl_item_fn.block,
        ) {
            Ok(blocking_fn) => blocking_fn,
            Err(err) => return err.into_compile_error().into(),
        };
        // return the regular function and the blocking function
        quote! {
            #impl_item_fn

            #blocking_fn
        }
        .into()
    } else {
        syn::Error::new(
            Span::call_site(),
            "`blocking_fn` attribute can only be applied functions",
        )
        .into_compile_error()
        .into()
    }
}

/// Create a blocking version of the function based on its elements
fn blocking_fn_helper(
    attrs: &[Attribute],
    vis: &Visibility,
    sig: &Signature,
    block: &Block,
) -> Result<proc_macro2::TokenStream, Error> {
    // make sure this is an async function
    if sig.asyncness.is_none() {
        return Err(Error::new(
            Span::call_site(),
            "`blocking_fn` attribute can only be applied to async functions",
        ));
    }
    // set the signature to be not async and add the '_blocking' suffix
    let mut new_sig = sig.clone();
    new_sig.asyncness = None;
    new_sig.ident = format_ident!("{}_blocking", new_sig.ident);
    let async_block = quote! {
        async {
            #block
        }
    };
    // return the regular function and the blocking function
    Ok(quote! {
        #(
            #attrs
        )*
        #vis #new_sig {
            RUNTIME.block_on(#async_block)
        }
    })
}
