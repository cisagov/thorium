//! Proc-macros for pyclasses

use std::collections::{HashMap, HashSet};

use darling::{FromMeta, util::PathList};
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{ToTokens, quote};
use syn::{
    Attribute, Error, Field, Fields, FieldsNamed, FieldsUnnamed, Ident, ItemStruct, LitStr, Type,
    Visibility, parse_quote, spanned::Spanned,
};

use crate::utils;

/// Arguments to the blocking subclient attribute
#[derive(Debug, Clone, Default, FromMeta)]
#[darling(derive_syn_parse, default, and_then = PyClassArgs::validate)]
struct PyClassArgs {
    /// Whether to clone the struct into a separate one with the 'Py'
    /// suffix, leaving the original struct unchanged
    clone: Option<PyClassClone>,
    /// Set `#[pyo3(get)]` for *all* public fields
    get_all: bool,
    /// Set `#[pyo3(set)]` for *all* public fields
    set_all: bool,
    /// Which fields the proc macro should automatically set `#[pyo3(get)]` for
    ///
    /// Cannot be used with `get_except`
    get: Option<PathList>,
    /// Which fields the proc macro should automatically set `#[pyo3(set)]` for
    ///
    /// Cannot be used with `set_except`
    set: Option<PathList>,
    /// Which fields the proc macro should *not* automatically set `#[pyo3(get)]` for,
    /// setting for all other fields
    ///
    /// Cannot be used with `get`
    get_except: Option<PathList>,
    /// Which fields the proc macro should *not* automatically set `#[pyo3(set)]` for,
    /// setting for all other fields
    ///
    /// Cannot be used with `set`
    set_except: Option<PathList>,
    /// The types within the struct to map to 'Py' types (with 'Py' suffix
    /// appended); requires those 'Py' types to have been created already using
    /// `pyclass(clone)` or `pyenum(clone)`; only applies when cloning the
    /// struct
    ///
    /// Necessary when a type needed to be modified for `PyO3` compatibility.
    /// The mapped type must implement From for the type it's mapping.
    pytypes: Vec<LitStr>,
}

#[derive(Debug, Clone, Default, FromMeta)]
#[darling(derive_syn_parse, default)]
struct PyClassClone {
    /// A list of traits to derive on the cloned class
    ///
    /// `Clone` is always derived because it's required for `pyclass`
    derive: Vec<LitStr>,
}

/// Returns whether the field should have the get/set attribute based on the args.
///
/// Precedence:
///   1. `*_all = true` = include every public field
///   2. `Some(list)` in allowlist = include only listed fields (must be public)
///   3. `Some(list)` in exceptlist = include every public field except listed
///   4. none of the above = don't include
macro_rules! is_get_or_set {
    (
        $all:expr,
        $allowlist:expr,
        $exceptlist:expr,
        $get_or_set_lit:literal,
        $field:expr $(,)?
    ) => {{
        // check if the field is public
        let is_public = matches!($field.vis, Visibility::Public(_));
        if $all {
            // get_all / set_all: every public field
            is_public
        } else if let Some(list) = &$allowlist {
            // explicit allowlist
            match &$field.ident {
                None => is_public,
                Some(ident) => {
                    if list.iter().any(|p| p.is_ident(ident)) {
                        if !is_public {
                            return Err(Error::new(
                                $field.span(),
                                format!(
                                    "#[{}] can only be applied to public fields! field '{}' is not public",
                                    $get_or_set_lit, ident
                                ),
                            ));
                        }
                        true
                    } else {
                        false
                    }
                }
            }
        } else if let Some(list) = &$exceptlist {
            // explicit denylist: all public fields except these
            match &$field.ident {
                None => is_public,
                Some(ident) => is_public && !list.iter().any(|p| p.is_ident(ident)),
            }
        } else {
            // nothing specified
            false
        }
    }};
}

/// Check whether all the `get/set/no_get/no_set` fields iterator are in the set of
/// all fields and return an error on the first field that isn't
macro_rules! check_field_in_get_set {
    ($args_iter:expr, $get_or_set_raw:literal, $all_idents:expr) => {
        for path in $args_iter {
            let Some(ident) = path.get_ident() else {
                return Err(Error::new(
                    path.span(),
                    format!(
                        "Invalid '{}' field: not a valid field name",
                        $get_or_set_raw
                    ),
                ));
            };
            if !$all_idents.contains(ident) {
                return Err(Error::new(
                    ident.span(),
                    format!(
                        "Invalid '{}' field: struct does not have field '{}'",
                        $get_or_set_raw, ident
                    ),
                ));
            }
        }
    };
}

impl PyClassArgs {
    /// Validate the args after parsing
    fn validate(self) -> darling::Result<Self> {
        if self.get.is_some() && self.get_except.is_some() {
            return Err(darling::Error::custom(
                "`get` and `get_except` are mutually exclusive",
            ));
        }
        if self.get_all && self.get_except.is_some() {
            return Err(darling::Error::custom(
                "`get_all` and `get_except` are mutually exclusive",
            ));
        }
        if self.get_all && self.get.is_some() {
            return Err(darling::Error::custom(
                "`get_all` and `get` are mutually exclusive",
            ));
        }
        if self.set_all && self.set_except.is_some() {
            return Err(darling::Error::custom(
                "`set_all` and `set_except` are mutually exclusive",
            ));
        }
        if self.set_all && self.set.is_some() {
            return Err(darling::Error::custom(
                "`set_all` and `set` are mutually exclusive",
            ));
        }
        if self.set.is_some() && self.set_except.is_some() {
            return Err(darling::Error::custom(
                "`set` and `set_except` are mutually exclusive",
            ));
        }
        Ok(self)
    }

    /// Returns a get and/or set pyo3 attribute for the given field based on the proc macro args
    ///
    /// # Arguments
    ///
    /// * `field` - The field to build the attribute for
    fn pyo3_field_attr(&self, field: &Field) -> Result<Option<Attribute>, Error> {
        let get = is_get_or_set!(self.get_all, self.get, self.get_except, "get", field);
        let set = is_get_or_set!(self.set_all, self.set, self.set_except, "set", field);
        match (get, set) {
            (true, true) => Ok(Some(parse_quote!(#[pyo3(get, set)]))),
            (true, false) => Ok(Some(parse_quote!(#[pyo3(get)]))),
            (false, true) => Ok(Some(parse_quote!(#[pyo3(set)]))),
            (false, false) => Ok(None),
        }
    }

    /// With the full set of field names in the struct, check to make sure the
    /// `get/set/no_get/no_set` args are valid
    ///
    /// # Arguments
    ///
    /// * `all_field_names` - All field names in the struct
    fn verify_get_set(&self, all_field_idents: &HashSet<&Ident>) -> Result<(), Error> {
        check_field_in_get_set!(
            self.get.as_deref().into_iter().flatten(),
            "get",
            all_field_idents
        );
        check_field_in_get_set!(
            self.set.as_deref().into_iter().flatten(),
            "set",
            all_field_idents
        );
        check_field_in_get_set!(
            self.get_except.as_deref().into_iter().flatten(),
            "get_except",
            all_field_idents
        );
        check_field_in_get_set!(
            self.set_except.as_deref().into_iter().flatten(),
            "set_except",
            all_field_idents
        );
        Ok(())
    }

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

/// Set the struct to be a `pyclass`, applying `get` and `set` attributes to
/// public fields depending on the given args
pub fn pyclass(args_raw: TokenStream, input: TokenStream) -> TokenStream {
    let orig_struct = match syn::parse::<ItemStruct>(input) {
        Ok(input_struct) => input_struct,
        Err(err) => {
            return Error::new(err.span(), "expected a struct")
                .to_compile_error()
                .into();
        }
    };
    let mut py_struct = orig_struct.clone();
    let args: PyClassArgs = match syn::parse(args_raw) {
        Ok(args) => args,
        Err(err) => return err.to_compile_error().into(),
    };
    // append the #[pyclass] attribute to the struct
    let pyclass_attr: Attribute = parse_quote!(#[pyclass(from_py_object)]);
    // try to add a `Clone` derive if one doesn't already exist
    py_struct.attrs.push(pyclass_attr);
    // add pyo3 attributes to each field depending on what we have set
    let mut all_field_names = HashSet::new();
    match &mut py_struct.fields {
        syn::Fields::Named(named) => {
            for field in &mut named.named {
                if let Some(ident) = &field.ident {
                    all_field_names.insert(ident);
                }
                // add a pyo3 field attribute for this field if needed
                match args.pyo3_field_attr(field) {
                    Ok(Some(attr)) => field.attrs.push(attr),
                    Ok(None) => (),
                    Err(err) => return err.into_compile_error().into(),
                }
            }
        }
        syn::Fields::Unnamed(unnamed) => {
            for field in &mut unnamed.unnamed {
                // add a pyo3 field attribute for this field if needed
                match args.pyo3_field_attr(field) {
                    Ok(Some(attr)) => field.attrs.push(attr),
                    Ok(None) => (),
                    Err(err) => return err.into_compile_error().into(),
                }
            }
        }
        syn::Fields::Unit => {}
    }
    // ensure that all get, set, get_except, and set_except are fields that this struct actually has
    if let Err(err) = args.verify_get_set(&all_field_names) {
        return err.into_compile_error().into();
    }
    let expanded = if let Some(clone_args) = &args.clone {
        let py_clone = pyclass_clone(&orig_struct, py_struct, &args, clone_args);
        quote! {
            #orig_struct

            #py_clone
        }
    } else {
        // just return the modified struct
        quote!(#py_struct)
    };
    expanded.into_token_stream().into()
}

/// Return a token stream containing the cloned python struct and any other
/// required items
fn pyclass_clone(
    orig_struct: &ItemStruct,
    mut py_struct: ItemStruct,
    args: &PyClassArgs,
    clone_args: &PyClassClone,
) -> proc_macro2::TokenStream {
    let orig_name = orig_struct.ident.to_string();
    let orig_name_lit = LitStr::new(&orig_name, py_struct.span());
    // add 'Py' suffix
    let py_name = format!("{orig_name}Py");
    py_struct.ident = Ident::new(&py_name, py_struct.ident.span());
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
    py_struct.attrs = py_attrs;
    let mut mapped_types = Vec::new();
    if !args.pytypes.is_empty() {
        let map_types = args.py_type_map();
        for field in &mut py_struct.fields {
            // save the old type before we overwrite it
            let old_type = field.ty.clone();
            if let Some(_new_ident) = utils::replace_type(&mut field.ty, &map_types) {
                // save both the old type and the new type to check if it's mapped
                // when generating from impls
                mapped_types.push(old_type);
                mapped_types.push(field.ty.clone());
            }
        }
    }
    let from_impls = pystruct_from_impls(orig_struct, &py_struct, &mapped_types);
    quote! {
        #py_struct

        #from_impls
    }
}

/// Generate the `From` implementations for the generated struct in both directions
fn pystruct_from_impls(
    orig_struct: &ItemStruct,
    py_struct: &ItemStruct,
    mapped_types: &[Type],
) -> proc_macro2::TokenStream {
    let orig_ident = &orig_struct.ident;
    let py_ident = &py_struct.ident;
    let v = Ident::new("v", py_struct.span());
    let from_orig_to_py = match &orig_struct.fields {
        Fields::Unnamed(fields) => map_struct_fields_unnamed(fields, mapped_types, &v, py_ident),
        Fields::Named(fields) => map_struct_fields_named(fields, mapped_types, &v, py_ident),
        Fields::Unit => {
            // trivial conversion
            quote! { #orig_ident => #py_ident, }
        }
    };
    let from_py_to_orig = match &py_struct.fields {
        Fields::Unnamed(fields) => map_struct_fields_unnamed(fields, mapped_types, &v, orig_ident),
        Fields::Named(fields) => map_struct_fields_named(fields, mapped_types, &v, orig_ident),
        Fields::Unit => {
            // trivial conversion
            quote! { #py_ident => #orig_ident, }
        }
    };
    quote! {
        impl From<#orig_ident> for #py_ident {
            fn from(#v: #orig_ident) -> Self {
                #from_orig_to_py
            }
        }

        impl From<#py_ident> for #orig_ident {
            fn from(#v: #py_ident) -> Self {
                #from_py_to_orig
            }
        }
    }
}

// Unnamed‑field (tuple) struct handling; adds `.into()` for mapped types
fn map_struct_fields_unnamed(
    fields: &FieldsUnnamed,
    mapped_types: &[Type],
    value_ident: &Ident,
    to_ident: &Ident,
) -> proc_macro2::TokenStream {
    // generate pattern identifiers p0, p1, …
    let binds = (0..fields.unnamed.len())
        .map(|i| {
            let i_ident = Ident::new(&i.to_string(), to_ident.span());
            let field_ty = &fields.unnamed[i].ty;
            if mapped_types.contains(field_ty) {
                // this type was mapped, so try to convert it
                let into = super::type_into(field_ty);
                quote! { #value_ident.#i_ident #into }
            } else {
                quote! { #value_ident.#i_ident }
            }
        })
        .collect::<Vec<_>>();
    let binds = quote! { #( #binds ),* };
    quote! {
        #to_ident( #binds )
    }
}

// Named‑field (regular) struct handling; adds `.into()` for mapped types
fn map_struct_fields_named(
    fields: &FieldsNamed,
    mapped_types: &[Type],
    value_ident: &Ident,
    to_ident: &Ident,
) -> proc_macro2::TokenStream {
    let binds = fields
        .named
        .iter()
        .filter_map(|field| {
            // all named fields will have this identity
            let ident = field.ident.as_ref()?;
            let bind = if mapped_types.contains(&field.ty) {
                // this type was mapped, so try to convert it
                let into = super::type_into(&field.ty);
                quote! { #ident: #value_ident.#ident #into }
            } else {
                quote! { #ident: #value_ident.#ident }
            };
            Some(bind)
        })
        .collect::<Vec<_>>();
    let construct = quote! { #( #binds ),* };
    quote! {
        #to_ident { #construct }
    }
}
