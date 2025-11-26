//! proc-macros for generating a blocking trait from an async one

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::{
    Error, Ident, ImplItemFn, ItemImpl, ItemTrait, Path, TraitItemFn, TypePath, parse_quote,
};

/// Create a wrapping, synchronous version of the given trait with the
/// suffix `Blocking` appended
///
/// The trait has a generic associated type (GAT) that implements the target
/// trait and a method that returns an instance of the GAT from `&self`.
/// Then all methods are just called on that inner GAT. Any asynchronous methods
/// are blocked on the static runtime to ensure the trait is synchronous.
///
/// The attribute can be applied either to a trait or to an impl for the trait.
/// When applied to a trait, a `Blocking` version of the trait is created. When
/// applied to an impl, the items in the impl are converted to synchronous versions
/// and added an impl for the `Blocking` trait.
pub fn blocking_trait(meta: TokenStream, input: TokenStream) -> TokenStream {
    // first check if this is a trait
    if let Ok(item_trait) = syn::parse::<ItemTrait>(input.clone()) {
        blocking_trait_trait(meta, item_trait)
    } else if let Ok(item_impl) = syn::parse::<ItemImpl>(input) {
        // is it an impl?
        blocking_trait_impl(meta, item_impl)
    } else {
        // return an error
        syn::Error::new(
            Span::call_site(),
            "`blocking_trait` attribute can only be applied to traits or trait impls",
        )
        .into_compile_error()
        .into()
    }
}

/// Create a wrapping, synchronous version of the given trait with the
/// suffix `Blocking` appended
fn blocking_trait_trait(_meta: TokenStream, item_trait: ItemTrait) -> TokenStream {
    // parse the input stream into our async function
    let trait_name = item_trait.ident;
    let blocking_trait_name = format_ident!("{}Blocking", trait_name);
    let trait_attrs = item_trait.attrs;
    let trait_vis = item_trait.vis;
    let generics = item_trait.generics;
    let supertraits = item_trait.supertraits;
    let items = item_trait.items;
    let mut blocking_items = Vec::new();
    // create an inner GAT that implements the trait we need; this is the inner
    // async client
    let inner_gat = quote! {
        type Inner: #trait_name;
    };
    blocking_items.push(inner_gat);
    // add a method declaration for getting the inner client that implements the
    // non-blocking trait
    let inner_method_name = inner_method_name(&trait_name);
    let inner_declaration = quote! {
        fn #inner_method_name(&self) -> &<Self as #blocking_trait_name>::Inner;
    };
    blocking_items.push(inner_declaration);
    for item in &items {
        match item {
            // if it's a default method, we need to provide an implementation
            syn::TraitItem::Fn(method) => match blocking_fn_trait(method, &inner_method_name) {
                Ok(blocking_method) => blocking_items.push(blocking_method),
                Err(err) => return err.into_compile_error().into(),
            },
            // otherwise leave it as is
            _ => {
                blocking_items.push(quote! { #item });
            }
        }
    }
    let supertraits_clause = if supertraits.is_empty() {
        quote! {}
    } else {
        quote! { : #supertraits }
    };
    let original_trait = quote! {
        #(
            #trait_attrs
        )*
        #trait_vis trait #trait_name #generics #supertraits_clause {
            #(
                #items
            )*
        }
    };
    let mut blocking_attrs = trait_attrs.clone();
    blocking_attrs.push(parse_quote! { #[cfg(feature = "sync")] });
    // generate the blocking trait, preserving the same supertrait bounds.
    let blocking_trait = quote! {
        #(
            #blocking_attrs
        )*
        #trait_vis trait #blocking_trait_name #generics {
            #(
                #blocking_items
            )*
        }
    };
    // return both the async and the sync trait
    let expanded = quote! {
        #original_trait

        #blocking_trait
    };
    expanded.into()
}

/// Construct a unique name for the inner method for the given trait
///
/// The inner method returns a reference to the inner field that implements
/// the non-blocking trait. We append the trait name as a suffix to ensure
/// the method is unique over multiple traits
fn inner_method_name(trait_ident: &Ident) -> Ident {
    let trait_lower = trait_ident.to_string().to_lowercase();
    format_ident!("inner_{trait_lower}")
}

/// Takes a trait method defined in the trait block and outputs a
/// synchronous version
///
/// If there's a default body, the method is called on the inner type. The inner
/// type is retrieved by calling the `inner()` trait method that reutnrs the async
/// client from `&self`.
fn blocking_fn_trait(
    method: &TraitItemFn,
    inner_method_name: &Ident,
) -> Result<proc_macro2::TokenStream, Error> {
    // check if this is an async method
    let is_async = method.sig.asyncness.is_some();
    // set the new signature to not async
    let mut new_sig = method.sig.clone();
    new_sig.asyncness = None;
    let method_attrs = &method.attrs;
    // only replace the body if there is a default body
    if method.default.is_some() {
        let method_name = &method.sig.ident;
        // collect argument identifiers (excluding the receiver)
        let mut arg_idents = Vec::new();
        for input in &method.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    arg_idents.push(pat_ident.ident.clone());
                } else {
                    return Err(Error::new(
                        Span::call_site(),
                        "Bad arg! Only typed args are supported",
                    ));
                }
            }
        }
        // build the argument list for the inner call
        let call_args = quote! { #(#arg_idents),* };
        // determine whether the method takes a receiver (self)
        let has_self = method
            .sig
            .inputs
            .iter()
            .any(|i| matches!(i, syn::FnArg::Receiver(_)));
        let target = if has_self {
            // call the trait method to get the inner instance that implements the non-blocking trait,
            // then call the respective method on that instance
            quote! { self.#inner_method_name().#method_name }
        } else {
            // call the associated async function directly on the type
            quote! { Self::#method_name }
        };
        // if the method is async, block it on the static runtime
        let block_body = if is_async {
            quote! {
                RUNTIME.block_on(#target(#call_args))
            }
        } else {
            quote! {
                #target(#call_args)
            }
        };
        // assemble the new method
        Ok(quote! {
            #(
                #method_attrs
            )*
            #new_sig {
                #block_body
            }
        })
    } else {
        // there is no default body, so just return the new signature (not async)
        Ok(quote! {
            #(
                #method_attrs
            )*
            #new_sig;
        })
    }
}

/// Create an impl block for the blocking trait
///
/// All methods are called on the inner type that implements the async trait. Any
/// async methods are blocked on the static runtime.
fn blocking_trait_impl(_meta: TokenStream, item_impl: ItemImpl) -> TokenStream {
    // get the self type for this impl
    let syn::Type::Path(self_ty) = *item_impl.self_ty else {
        return Error::new_spanned(&item_impl.self_ty, "Only type paths are supported")
            .to_compile_error()
            .into();
    };
    // get the methods implemented in this impl
    let items = &item_impl.items;
    // get attributes (docstrings/examples) for our function
    let attrs = &item_impl.attrs;
    // get the trait that this impl is implementing
    let Some(trait_path) = item_impl.trait_.as_ref().map(|(_, path, _)| path) else {
        return Error::new(
            Span::call_site(),
            "The `blocking_trait` attribute can only be applied to trait impls",
        )
        .to_compile_error()
        .into();
    };
    // get the trait's identity
    let Some(trait_ident) = trait_path.get_ident() else {
        return Error::new(Span::call_site(), "Unable to get trait identity")
            .to_compile_error()
            .into();
    };
    // build sync name
    let ident = self_ty.path.get_ident().unwrap();
    let sync_name = syn::Ident::new(&format!("{ident}Blocking"), ident.span());
    // get information on the generics to pass
    let generics = &item_impl.generics;
    let mut blocking_items = Vec::new();
    // add a type for our inner that implements all necessary traits
    let inner_associated_type = quote! {
        type Inner = #self_ty;
    };
    blocking_items.push(inner_associated_type);
    // define the inner method that actually gets the inner field that implements
    // the non-blocking trait
    let inner_method_name = inner_method_name(trait_ident);
    let inner_definition = quote! {
        fn #inner_method_name(&self) -> &#self_ty {
            &self.inner
        }
    };
    blocking_items.push(inner_definition);
    // look for any async methods that we need to convert
    for item in items {
        match item {
            // if this is a method, call the method on the inner client
            syn::ImplItem::Fn(method) => {
                match blocking_impl_fn_trait(method, &inner_method_name, trait_path, &self_ty) {
                    Ok(blocking_method) => blocking_items.push(blocking_method),
                    Err(err) => return err.into_compile_error().into(),
                }
            }
            // otherwise leave it as is
            _ => {
                blocking_items.push(quote! { #item });
            }
        }
    }
    // build async impl
    let async_impl = quote! {
        #(#attrs)*
        #generics
        impl #trait_path for #self_ty {
            #(#items)*
        }
    };
    // build sync impl
    let blocking_trait_name = syn::Ident::new(&format!("{trait_ident}Blocking"), ident.span());
    let sync_impl = quote! {
        #[cfg(feature = "sync")]
        impl #blocking_trait_name for #sync_name {
            #(
                #blocking_items
            )*
        }
    };
    // combine both impl blocks into the final token stream
    let output = quote! {
        #async_impl

        #sync_impl
    };
    output.into()
}

/// Takes a method within an impl block for a trait and just sets the body to calling
/// the method on the inner async client
///
/// The async client (associated type) is retrieved by using the inner trait method.
/// If the method is async, the method is called by blocking on the static runtime.
fn blocking_impl_fn_trait(
    method: &ImplItemFn,
    inner_method_name: &Ident,
    trait_path: &Path,
    self_ty: &TypePath,
) -> Result<proc_macro2::TokenStream, Error> {
    // check if this is an async method
    let is_async = method.sig.asyncness.is_some();
    // set the new signature to not async
    let mut new_sig = method.sig.clone();
    new_sig.asyncness = None;
    let vis = &method.vis;
    let method_attrs = &method.attrs;
    // Collect argument identifiers (excluding the receiver)
    let mut arg_idents = Vec::new();
    for input in &method.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                arg_idents.push(pat_ident.ident.clone());
            } else {
                return Err(Error::new(
                    Span::call_site(),
                    "Bad arg! Only typed args are supported",
                ));
            }
        }
    }
    // build the argument list for the inner call
    let call_args = quote! { #(#arg_idents),* };
    let method_name = &method.sig.ident;
    // determine whether the method takes a receiver (self)
    let has_self = method
        .sig
        .inputs
        .iter()
        .any(|i| matches!(i, syn::FnArg::Receiver(_)));
    let target = if has_self {
        // Call the method on the inner instance
        quote! { self.#inner_method_name().#method_name }
    } else {
        // Call the associated async function directly on the type
        quote! { <#self_ty as #trait_path>::#method_name }
    };
    // if the method is async, block it on the static runtime
    let block_body = if is_async {
        quote! {
            RUNTIME.block_on(#target(#call_args))
        }
    } else {
        quote! {
            #target(#call_args)
        }
    };
    // assemble the new method
    Ok(quote! {
        #(
            #method_attrs
        )*
        #vis #new_sig {
            #block_body
        }
    })
}
