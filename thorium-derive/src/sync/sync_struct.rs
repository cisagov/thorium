//! proc-macros for generating a blocking struct from an async one

use std::collections::HashMap;

use darling::FromMeta;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::{
    Attribute, Error, Expr, Fields, Generics, Ident, ImplItemFn, ItemImpl, ItemStruct, Lit, LitStr,
    Meta, ReturnType, Signature, Type, TypePath, parse_quote,
};

use crate::utils;

/// Arguments to the blocking struct attribute
#[derive(Debug, Clone, Default, FromMeta)]
#[darling(derive_syn_parse, default)]
struct BlockingStructArgs {
    /// The list of fields to provide getters that return references to the inner
    /// async struct's fields.
    ///
    /// Since the inner struct is private, this is useful if the async struct
    /// doesn't provide function getters and you need access to its data.
    getters: Vec<LitStr>,
    /// The list of return types to automatically wrap
    ///
    /// This is useful when a function returns an async struct and you want to
    /// automatically wrap it into a blocking struct. Requires the wrapping type
    /// to already be defined and to implement `From` for the type it's wrapping.
    wrap_return: HashMap<Ident, Ident>,
    /// Whether the proc macro should automatically set ``PyO3``
    /// attributes
    python: bool,
}

/// Creates a blocking client by wrapping the async client and calling its
/// async methods by blocking on a static runtime.
///
/// When applied to a struct, this creates a the blocking client wrapper with the
/// suffix `Blocking` added.
///
/// When applied to an impl block, it duplicates all of the async client's methods,
/// and sets the body to just calling the method on the inner async client. Async
/// methods are blocked on the static runtime.
pub fn blocking_struct(args_raw: TokenStream, input: TokenStream) -> TokenStream {
    let args = match syn::parse(args_raw) {
        Ok(args) => args,
        Err(err) => return err.to_compile_error().into(),
    };
    // first check if this is a struct
    if let Ok(item_struct) = syn::parse::<ItemStruct>(input.clone()) {
        blocking_struct_struct(&item_struct, &args)
    } else if let Ok(item_impl) = syn::parse::<ItemImpl>(input) {
        // is it an impl?
        blocking_struct_impl(item_impl, &args)
    } else {
        // return an error
        syn::Error::new(
            Span::call_site(),
            "`blocking_struct` attribute can only be applied to structs or struct impls",
        )
        .into_compile_error()
        .into()
    }
}

/// Create a blocking version of the struct by wrapping the async struct
fn blocking_struct_struct(item_struct: &ItemStruct, args: &BlockingStructArgs) -> TokenStream {
    let struct_name = &item_struct.ident;
    let blocking_struct_name = format_ident!("{}Blocking", struct_name);
    let struct_attrs = &item_struct.attrs;
    let struct_vis = &item_struct.vis;
    let generics = &item_struct.generics;
    let (impl_gen, ty_gen, where_clause) = generics.split_for_impl();
    // the original struct is just what we got in
    let original_struct = quote! {
        #item_struct
    };
    let mut blocking_attrs = struct_attrs.clone();
    blocking_attrs.push(parse_quote! { #[cfg(feature = "sync")] });
    if args.python {
        // make this blocking struct a pyclass if python is set
        blocking_attrs
            .push(parse_quote! { #[cfg_attr(feature = "python", pyclass(from_py_object))] });
    }
    // the blocking struct wraps the original struct with
    // the same attributes
    let blocking_struct = quote! {
        #(
            #blocking_attrs
        )*
        #struct_vis struct #blocking_struct_name #impl_gen #where_clause {
            inner: #struct_name #ty_gen
        }
    };
    // provide getters if any were requested
    let getters_impl = match getters_impl(
        &args.getters,
        &item_struct.fields,
        &blocking_struct_name,
        generics,
    ) {
        Ok(getters) => getters,
        Err(err) => return err.into_compile_error().into(),
    };
    // provide a from implementation for the blocking struct
    let from_impl = quote! {
        impl #impl_gen From<#struct_name #ty_gen> for #blocking_struct_name #ty_gen
        #where_clause
        {
            fn from(inner: #struct_name #ty_gen) -> Self {
                Self { inner }
            }
        }
    };
    // combine everything.
    let expanded = quote! {
        #original_struct

        #blocking_struct

        #from_impl

        #getters_impl
    };
    expanded.into()
}

/// Create a blocking impl for the blocking client wrapper by having methods just
/// call the corresponding method on the inner async client. If the method is async,
/// the method is blocked on the static runtime.
fn blocking_struct_impl(item_impl: ItemImpl, args: &BlockingStructArgs) -> TokenStream {
    // get the methods implemented in this impl
    let items = &item_impl.items;
    // get attributes (docstrings/examples) for our function
    let attrs = &item_impl.attrs;
    // get the self type for this impl
    let syn::Type::Path(self_ty) = *item_impl.self_ty else {
        return syn::Error::new_spanned(&item_impl.self_ty, "Only type paths are supported")
            .to_compile_error()
            .into();
    };
    // get the optional trait that this impl might be implementing
    if item_impl.trait_.is_some() {
        return syn::Error::new(
            Span::call_site(),
            "Trait impls are not supported for `blocking_struct`. \
            Use `blocking_trait` instead",
        )
        .to_compile_error()
        .into();
    }
    // build sync name
    let ident = self_ty
        .path
        .segments
        .last()
        .expect("type path must have at least one segment")
        .ident
        .clone();
    let sync_name = syn::Ident::new(&format!("{ident}Blocking"), ident.span());
    // get information on the generics to pass
    let generics = &item_impl.generics;
    let (impl_gen, ty_gen, where_clause) = generics.split_for_impl();
    let mut blocking_items = Vec::new();
    // look for any async methods that we need to convert
    for item in items {
        match item {
            // if this is a method, call the method on the inner client
            syn::ImplItem::Fn(method) => {
                if is_new(&method.sig) {
                    // this is a 'new' method, so create a new method for the blocking struct
                    match blocking_new(method, &self_ty) {
                        Ok(blocking_method) => blocking_items.push(blocking_method),
                        Err(err) => return err.into_compile_error().into(),
                    }
                } else {
                    // otherwise, try to convert the method
                    match blocking_impl_fn(method, &self_ty, args) {
                        Ok(blocking_method) => blocking_items.push(blocking_method),
                        Err(err) => return err.into_compile_error().into(),
                    }
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
        impl #impl_gen  #ident #ty_gen
        #where_clause
        {
            #(#items)*
        }
    };
    let mut blocking_attrs: Vec<Attribute> = vec![parse_quote! { #[cfg(feature = "sync")] }];
    if args.python {
        blocking_attrs.push(parse_quote! { #[cfg_attr(feature = "python", pymethods)] });
    }
    // build sync impl
    let sync_impl = quote! {
        #(
            #blocking_attrs
        )*
        impl #impl_gen  #sync_name #ty_gen
        #where_clause
        {
            #(
                #blocking_items
            )*
        }
    };
    // Combine both impl blocks into the final token stream
    let output = quote! {
        #async_impl

        #sync_impl
    };
    output.into()
}

/// Returns an impl block containing the requested getters
fn getters_impl(
    getter_fields: &[LitStr],
    fields: &Fields,
    sync_name: &Ident,
    generics: &Generics,
) -> Result<proc_macro2::TokenStream, Error> {
    if getter_fields.is_empty() {
        return Ok(quote! {});
    }
    let (impl_gen, ty_gen, where_clause) = generics.split_for_impl();
    let mut getters = Vec::new();
    // add getters if any were requested
    for getter_field in getter_fields {
        let getter_string = getter_field.value();
        if let Some(field) = fields.iter().find(|field| {
            field
                .ident
                .as_ref()
                .is_some_and(|ident| *ident == getter_string)
        }) {
            let field_ty = &field.ty;
            let getter_ident = Ident::new(&getter_string, Span::call_site());
            // add a getter for this field
            getters.push(quote! {
                pub fn #getter_ident(&self) -> &#field_ty {
                    &self.inner.#getter_ident
                }
            });
        } else {
            // return an error if the struct has no such field
            return Err(Error::new(
                Span::call_site(),
                format!("invalid getter field: struct has no such field '{getter_string}'"),
            ));
        }
    }
    Ok(quote! {
        impl #impl_gen #sync_name #ty_gen
        #where_clause
        {
            #(
                #getters
            )*
        }
    })
}

/// Creates a `new` method for the blocking struct based on
/// the async `new` method
///
/// The blocking `new` just wraps an instance of the async struct
fn blocking_new(new: &ImplItemFn, self_ty: &TypePath) -> Result<proc_macro2::TokenStream, Error> {
    let mut sig = new.sig.clone();
    let vis = &new.vis;
    let method_attrs = trim_doc_examples(&new.attrs);
    let method_name = &new.sig.ident;
    let mut args = Vec::new();
    for input in &new.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                args.push(pat_ident.ident.clone());
            } else {
                return Err(Error::new(
                    Span::call_site(),
                    "Bad arg! Only typed args are supported",
                ));
            }
        }
    }
    // check if this is an async method
    let is_async = new.sig.asyncness.is_some();
    // check if it returns a result (is fallible)
    let returns_result = return_type_matches(&new.sig.output, "Result");
    // the body should just return an instance of the blocking struct wrapping an
    // instance of the async struct
    let body = match (is_async, returns_result) {
        (true, true) => {
            quote! { Ok(Self { inner: RUNTIME.block_on(<#self_ty>::#method_name(#(#args),*))? }) }
        }
        (true, false) => {
            quote! { Self { inner: RUNTIME.block_on(<#self_ty>::#method_name(#(#args),*)) }}
        }
        (false, true) => quote! { Ok(Self { inner: <#self_ty>::#method_name(#(#args),*)? }) },
        (false, false) => quote! { Self { inner: <#self_ty>::#method_name(#(#args),*) }},
    };
    // ensure the final method is not async
    sig.asyncness = None;
    // assemble the new method
    Ok(quote! {
        #(
            #method_attrs
        )*
        #vis #sig {
            #body
        }
    })
}

/// Remove the examples section and everything following from the
/// doc comments in the given list of attributes
fn trim_doc_examples(attrs: &[Attribute]) -> Vec<Attribute> {
    if let Some(examples_start) = attrs.iter().position(|attr| {
        if attr.path().is_ident("doc")
            && let Meta::NameValue(meta_name_value) = &attr.meta
            && let Expr::Lit(expr_lit) = &meta_name_value.value
            && let Lit::Str(lit_str) = &expr_lit.lit
        {
            return lit_str.value().contains("# Examples");
        }
        false
    }) {
        // get everything before the examples
        let before_examples = &attrs[..examples_start];
        // skip the rest of the docs
        let after_examples = attrs[examples_start..]
            .iter()
            .skip_while(|attr| attr.path().is_ident("doc"))
            .cloned()
            .collect::<Vec<_>>();
        // combine the two parts into a new Vec
        let mut new_attrs = Vec::new();
        new_attrs.extend_from_slice(before_examples);
        new_attrs.extend(after_examples);
        new_attrs
    } else {
        attrs.to_vec()
    }
}

/// Takes an impl method and just sets the body to calling the method
/// on the inner async client. If the method is async, the method is
/// called by blocking on the static runtime.
///
/// Detects `new` functions and generates a new function that creates
/// the async struct and wraps it in the blocking struct
fn blocking_impl_fn(
    method: &ImplItemFn,
    self_ty: &TypePath,
    args: &BlockingStructArgs,
) -> Result<proc_macro2::TokenStream, Error> {
    // check if this is an async method
    let is_async = method.sig.asyncness.is_some();
    // set the new signature to not async
    let mut new_sig = method.sig.clone();
    new_sig.asyncness = None;
    let vis = &method.vis;
    let method_attrs = trim_doc_examples(&method.attrs);
    let method_name = &method.sig.ident;
    let maybe_wrapped_ident = wrap_return(&mut new_sig, &args.wrap_return);
    // determine whether the method takes a receiver (self)
    let has_self = method
        .sig
        .inputs
        .iter()
        .any(|i| matches!(i, syn::FnArg::Receiver(_)));
    // determine whether method returns `Result`
    let returns_result = return_type_matches(&method.sig.output, "Result");
    // determine whether the method returns `Self`
    let returns_self = return_type_matches(&method.sig.output, "Self");
    let target = if has_self {
        // call the method on the inner instance
        quote! { self.inner.#method_name }
    } else {
        // call the associated async function directly on the type
        quote! { #self_ty::#method_name }
    };
    let mut args = Vec::new();
    for input in &method.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                args.push(pat_ident.ident.clone());
            } else {
                return Err(Error::new(
                    Span::call_site(),
                    "Bad arg! Only typed args are supported",
                ));
            }
        }
    }
    // Build the argument list for the inner call
    let call_args = quote! { #(#args),* };
    let block_body = if is_async {
        // if the method is async, block it on the static runtime
        quote! {
            RUNTIME.block_on(#target(#call_args))
        }
    } else {
        // otherwise just call the function on the inner
        quote! {
            #target(#call_args)
        }
    };
    let block_body = if let Some(wrapped_ident) = maybe_wrapped_ident {
        // we wrapped the return type, so convert it using the wrapping type's From implementation
        if returns_result {
            // we're returning a result, so make sure we also return one
            quote! { Ok(#wrapped_ident::from(#block_body?)) }
        } else {
            quote! { #wrapped_ident::from(#block_body) }
        }
    } else if returns_self {
        // method returns `Self`, so set inner to output and return self
        quote! {
            let inner = #block_body;
            self.inner = inner;
            self
        }
    } else {
        block_body
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

/// Wrap the return type if it's in the wrap map
///
/// Returns a reference to the wrapping identity if one was found
fn wrap_return<'a>(
    new_sig: &mut Signature,
    wrap_return: &'a HashMap<Ident, Ident>,
) -> Option<&'a Ident> {
    // exit early if we have no return types to wrap, avoiding unnecessary recursion
    if wrap_return.is_empty() {
        return None;
    }
    let r_ty = match &mut new_sig.output {
        ReturnType::Default => return None,
        ReturnType::Type(_, r_ty) => r_ty,
    };
    // look for types to replace recursively, only replacing the first one we find
    utils::replace_type(r_ty, wrap_return)
}

/// Checks if the function is a `new` function by naively checking
/// its identity and that it does not take a receiver
fn is_new(sig: &Signature) -> bool {
    sig.ident == "new"
        && sig
            .inputs
            .iter()
            .all(|i| !matches!(i, syn::FnArg::Receiver(_)))
}

/// Checks that the return type is literally equivalent to the given raw str
pub fn return_type_matches(ret: &ReturnType, type_raw: &str) -> bool {
    let ret_ty = match ret {
        ReturnType::Default => return false,
        ReturnType::Type(_, ty) => ty,
    };
    let Type::Path(TypePath {
        qself: None | Some(_),
        path,
    }) = ret_ty.as_ref()
    else {
        return false;
    };
    match path.segments.last() {
        // check if the identity is literally "Result"
        Some(seg) => seg.ident == type_raw,
        None => false,
    }
}
