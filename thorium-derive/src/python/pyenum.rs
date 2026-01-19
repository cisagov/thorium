//! Proc-macros for pyenums

use std::collections::HashMap;

use darling::FromMeta;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{ToTokens, quote};
use syn::{
    Error, Fields, FieldsNamed, FieldsUnnamed, Ident, ItemEnum, LitStr, Type, parse_quote,
    punctuated::Punctuated, token::Paren,
};

use crate::utils;

/// Arguments to the blocking subclient attribute
#[derive(Debug, Clone, Default, FromMeta)]
#[darling(derive_syn_parse, default)]
struct PyEnumArgs {
    /// Whether to clone the struct into a separate one with the 'Py'
    /// suffix, leaving the original struct unchanged
    ///
    /// This is required for complex enums with unit types
    clone: Option<PyEnumClone>,
    /// The types within the struct to map to 'Py' types (with 'Py' suffix
    /// appended); requires those 'Py' types to have been created already using
    /// `pyclass(clone)` or `pyenum(clone)`
    ///
    /// Necessary when a type needed to be modified for `PyO3` compatibility.
    /// The mapped type must implement From for the type it's mapping.
    pytypes: Vec<LitStr>,
    /// Rename variants in the cloned 'Py' enum
    rename: HashMap<Ident, Ident>,
}

#[derive(Debug, Clone, Default, FromMeta)]
#[darling(derive_syn_parse, default)]
struct PyEnumClone {
    /// A list of traits to derive on the cloned class
    ///
    /// `Clone` is always derived because it's required for `pyclass`
    derive: Vec<LitStr>,
}

impl PyEnumArgs {
    /// Returns a map of the type to map to the py type it should
    /// map to
    fn py_type_map(&self) -> HashMap<Ident, Ident> {
        self.pytypes
            .iter()
            .map(|ty_raw| {
                let ty_str = ty_raw.value();
                let ty_py_str = format!("{ty_str}Py");
                (
                    Ident::new(&ty_str, Span::call_site()),
                    Ident::new(&ty_py_str, Span::call_site()),
                )
            })
            .collect()
    }
}

pub fn pyenum(args_raw: TokenStream, input: TokenStream) -> TokenStream {
    let orig_enum = match syn::parse::<ItemEnum>(input) {
        Ok(e) => e,
        Err(err) => {
            return Error::new(err.span(), "expected an enum")
                .to_compile_error()
                .into();
        }
    };
    let args: PyEnumArgs = match syn::parse(args_raw) {
        Ok(args) => args,
        Err(err) => return err.to_compile_error().into(),
    };
    let expanded = if let Some(clone_args) = &args.clone {
        let py_enum = orig_enum.clone();
        pyenum_clone(&orig_enum, py_enum, &args, clone_args)
    } else {
        quote!(#orig_enum)
    };

    expanded.into_token_stream().into()
}

/// Defines an identical enum with the suffix `Py` appended that's PyO3-compatible,
/// replacing unit-type variants (e.g. `MyVariant`) with empty tuple variants
/// (e.g. `MyVariant()`).
///
/// From is implemented either side to easily convert types between each other. No
/// attributes are carried over.
///
/// This is required until `PyO3` supports unit-type enum variants in complex enums
/// (see <https://pyo3.rs/v0.27.2/class.html#complex-enums>)
fn pyenum_clone(
    orig_enum: &ItemEnum,
    mut py_enum: ItemEnum,
    args: &PyEnumArgs,
    clone_args: &PyEnumClone,
) -> proc_macro2::TokenStream {
    let orig_name = orig_enum.ident.to_string();
    let orig_name_lit = LitStr::new(&orig_name, py_enum.ident.span());
    // add 'Py' suffix
    let py_name = format!("{orig_name}Py");
    py_enum.ident = Ident::new(&py_name, py_enum.ident.span());
    let mut py_attrs = Vec::new();
    // add the #[pyclass] attribute required by PyO3; retain the original name for Python purposes
    py_attrs.push(parse_quote!(#[pyclass(name = #orig_name_lit, from_py_object)]));
    // add derive attributes
    let mut derive = clone_args
        .derive
        .iter()
        .map(|derive_str| Ident::new(&derive_str.value(), Span::call_site()))
        .collect::<Vec<_>>();
    // `Clone` is required for Python objects that implement `FromPyObject`
    derive.push(Ident::new("Clone", Span::call_site()));
    py_attrs.push(parse_quote!(#[derive( #( #derive ),* )]));

    py_enum.attrs = py_attrs;
    let mut tupled_variants = Vec::new();
    let mut mapped_types = Vec::new();
    // turn unit variants (`Foo`) into empty‑tuple variants (`Foo()`)
    for variant in &mut py_enum.variants {
        // first rename the variant if it's in the rename map
        if let Some(new_variant) = args.rename.get(&variant.ident) {
            variant.ident = new_variant.clone();
        }
        // make it an empty tuple variant if it's a unit variant
        if matches!(variant.fields, Fields::Unit) {
            tupled_variants.push(variant.ident.clone());
            variant.fields = Fields::Unnamed(FieldsUnnamed {
                paren_token: Paren::default(),
                unnamed: Punctuated::new(),
            });
        }
        // map any requested types (`Foo` -> `FooPy`)
        if !args.pytypes.is_empty() {
            let map_types = args.py_type_map();
            for field in &mut variant.fields {
                // make a copy of the old type to keep track of what was mapped
                let old_type = field.ty.clone();
                if let Some(_new_type) = utils::replace_type(&mut field.ty, &map_types) {
                    // push both the new type and the old one
                    mapped_types.push(field.ty.clone());
                    mapped_types.push(old_type);
                }
            }
        }
    }
    let from_impls = pyenum_from_impls(
        &py_enum,
        orig_enum,
        &tupled_variants,
        &mapped_types,
        &args.rename,
    );
    quote! {
        #orig_enum

        #py_enum

        #from_impls
    }
}

/// Generate the `From` implementations for the generated enum in both
/// directions
fn pyenum_from_impls(
    py_enum: &ItemEnum,
    orig_enum: &ItemEnum,
    tupled_variants: &[Ident],
    mapped_types: &[Type],
    rename: &HashMap<Ident, Ident>,
) -> proc_macro2::TokenStream {
    let orig_ident = &orig_enum.ident;
    let py_ident = &py_enum.ident;
    // track the reverse of variants we're renaming to retrieve them when going from the other direction
    let rename_reverse = rename
        .iter()
        .map(|(name, new_name)| (new_name, name))
        .collect::<HashMap<_, _>>();
    let from_orig_to_py = orig_enum.variants.iter().map(|v| {
        let name = &v.ident;
        // get the new name if it was renamed, otherwise just use the old one
        let new_name = rename.get(name).unwrap_or(name);
        match &v.fields {
            Fields::Unnamed(fields) => {
                map_enum_fields_unnamed(fields, mapped_types, orig_ident, py_ident, name, new_name)
            }
            Fields::Named(fields) => {
                map_enum_fields_named(fields, mapped_types, orig_ident, py_ident, name, new_name)
            }
            // unit variant becomes an empty‑tuple `()`
            Fields::Unit => {
                quote! {
                    #orig_ident::#name => #py_ident::#new_name(),
                }
            }
        }
    });
    let from_py_to_orig = py_enum.variants.iter().map(|v| {
        let name = &v.ident;
        // get the original name if it was renamed, otherwise just use the "new" one
        let orig_name = rename_reverse.get(name).unwrap_or(&name);
        match &v.fields {
            Fields::Unnamed(fields) => {
                if tupled_variants.contains(&v.ident) {
                    // this is a variant we changed to an empty tuple, so handle that
                    quote! {
                        #py_ident::#name() => #orig_ident::#orig_name,
                    }
                } else {
                    map_enum_fields_unnamed(
                        fields,
                        mapped_types,
                        py_ident,
                        orig_ident,
                        name,
                        orig_name,
                    )
                }
            }
            // Struct‑like – forward each named field
            Fields::Named(fields) => {
                map_enum_fields_named(fields, mapped_types, py_ident, orig_ident, name, orig_name)
            }
            // we never generate a unit variant in the Py enum
            Fields::Unit => unreachable!(),
        }
    });
    quote! {
        impl From<#orig_ident> for #py_ident {
            fn from(v: #orig_ident) -> Self {
                match v {
                    #( #from_orig_to_py )*
                }
            }
        }

        impl From<#py_ident> for #orig_ident {
            fn from(v: #py_ident) -> Self {
                match v {
                    #( #from_py_to_orig )*
                }
            }
        }
    }
}

/// Map the unnamed enum fields, ensuring any required conversions are handled
fn map_enum_fields_unnamed(
    fields: &FieldsUnnamed,
    mapped_types: &[Type],
    from_ident: &Ident,
    to_ident: &Ident,
    from_name: &Ident,
    to_name: &Ident,
) -> proc_macro2::TokenStream {
    // generate pattern identifiers p0, p1, …
    let (pats, binds): (Vec<_>, Vec<_>) = (0..fields.unnamed.len())
        .map(|i| {
            let id_str = format!("p{i}");
            let pat = Ident::new(&id_str, from_name.span());
            let field_ty = &fields.unnamed[i].ty;
            let bind = if mapped_types.contains(field_ty) {
                let into = super::type_into(field_ty);
                quote! { #pat #into }
            } else {
                quote! { #pat }
            };
            (pat, bind)
        })
        .unzip();
    let pats = quote! { #( #pats ),* };
    let binds = quote! { #( #binds ),* };
    quote! {
        #from_ident::#from_name( #pats ) => #to_ident::#to_name( #binds ),
    }
}

/// Map the named enum fields, ensuring any required conversions are handled
fn map_enum_fields_named(
    fields: &FieldsNamed,
    mapped_types: &[Type],
    orig_ident: &Ident,
    py_ident: &Ident,
    from_name: &Ident,
    to_name: &Ident,
) -> proc_macro2::TokenStream {
    let (pats, binds): (Vec<_>, Vec<_>) = fields
        .named
        .iter()
        .filter_map(|field| {
            // all named fields will have an ident
            let ident = field.ident.as_ref()?;
            let pat = ident.clone();
            let bind = if mapped_types.contains(&field.ty) {
                let into = super::type_into(&field.ty);
                quote! { #ident: #ident #into }
            } else {
                quote! { #ident }
            };
            Some((pat, bind))
        })
        .unzip();
    let pat = quote! { { #( #pats ),* } };
    let construct = quote! { { #( #binds ),* } };
    quote! {
        #orig_ident::#from_name #pat => #py_ident::#to_name #construct,
    }
}
