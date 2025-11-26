extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::Ident;

#[cfg(feature = "sync")]
mod sync;

#[cfg(feature = "python")]
mod python;

/// Add the json based serialzie impl
fn add_json_serialize(stream: &mut proc_macro2::TokenStream, name: &Ident) {
    // extend our token stream
    stream.extend(quote! {
        impl scylla::serialize::value::SerializeValue for #name {
            fn serialize<'b>(
                &self,
                typ: &scylla::frame::response::result::ColumnType,
                writer: scylla::serialize::writers::CellWriter<'b>,
            ) -> Result<scylla::serialize::writers::WrittenCellProof<'b>, scylla::serialize::SerializationError> {
                // cast our tag kind as a str
                let value = match serde_json::to_string(self) {
                    Ok(value) => value,
                    Err(error) => return Err(scylla::serialize::SerializationError::new(error)),
                };
                // serialize our tag kind
                scylla::serialize::value::SerializeValue::serialize(&value, typ, writer)
            }
        }
    })
}

/// Add the json based serialzie impl
fn add_json_deserialize(stream: &mut proc_macro2::TokenStream, name: &Ident) {
    // extend our token stream
    stream.extend(quote! {
        impl<'frame, 'metadata> scylla::deserialize::value::DeserializeValue<'frame, 'metadata> for #name {
            fn type_check(typ: &scylla::frame::response::result::ColumnType) -> Result<(), scylla::deserialize::TypeCheckError> {
                if let scylla::frame::response::result::ColumnType::Native(scylla::cluster::metadata::NativeType::Text) = typ {
                    return Ok(());
                }
                Err(scylla::deserialize::TypeCheckError::new(crate::models::scylla_utils::errors::DeserializationError::ExpectedText))
            }

            fn deserialize(
                _typ: &'metadata scylla::frame::response::result::ColumnType<'metadata>,
                v: Option<scylla::deserialize::FrameSlice<'frame>>,
            ) -> Result<Self, scylla::deserialize::DeserializationError> {
                // check if we got data
                match v {
                    Some(fslice) => {
                        // get the correct value
                        match serde_json::from_slice(fslice.as_slice()) {
                            Ok(event_type) => Ok(event_type),
                            Err(_) => Err(scylla::deserialize::DeserializationError::new(
                                crate::models::scylla_utils::errors::DeserializationError::UnknownValue,
                            )),
                        }
                    }
                    None => Err(scylla::deserialize::DeserializationError::new(
                        crate::models::scylla_utils::errors::DeserializationError::ExpectedNotNull,
                    )),
                }
            }
        }

    })
}

#[proc_macro_derive(ScyllaStoreJson)]
pub fn derive_scylla_store_json(stream: TokenStream) -> TokenStream {
    // parse our input struct
    let ast = syn::parse_macro_input!(stream as syn::DeriveInput);
    // get the name of our ident
    let name = &ast.ident;
    // start with an empty stream
    let mut output = quote! {};
    // add our json derives
    add_json_serialize(&mut output, name);
    add_json_deserialize(&mut output, name);
    output.into()
}

/// Add the as str based serialzie impl
fn add_as_str_serialize(stream: &mut proc_macro2::TokenStream, name: &Ident) {
    // extend our token stream
    stream.extend(quote! {
        impl scylla::serialize::value::SerializeValue for #name {
            fn serialize<'b>(
                &self,
                typ: &scylla::frame::response::result::ColumnType,
                writer: scylla::serialize::writers::CellWriter<'b>,
            ) -> Result<scylla::serialize::writers::WrittenCellProof<'b>, scylla::serialize::SerializationError> {
                // cast our value as a str
                let value = &self.as_str();
                // serialize our tag kind
                scylla::serialize::value::SerializeValue::serialize(&value, typ, writer)
            }
        }
    })
}

/// Add the as str based serialzie impl
fn add_as_str_deserialize(stream: &mut proc_macro2::TokenStream, name: &Ident) {
    // extend our token stream
    stream.extend(quote! {
        impl<'frame, 'metadata> scylla::deserialize::value::DeserializeValue<'frame, 'metadata> for #name {
            fn type_check(typ: &scylla::frame::response::result::ColumnType) -> Result<(), scylla::deserialize::TypeCheckError> {
                if let scylla::frame::response::result::ColumnType::Native(scylla::cluster::metadata::NativeType::Text) = typ {
                    return Ok(());
                }
                Err(scylla::deserialize::TypeCheckError::new(crate::models::scylla_utils::errors::DeserializationError::ExpectedText))
            }

            fn deserialize(
                _typ: &'metadata scylla::frame::response::result::ColumnType<'metadata>,
                v: Option<scylla::deserialize::FrameSlice<'frame>>,
            ) -> Result<Self, scylla::deserialize::DeserializationError> {
                // check if we got data
                match v {
                    Some(fslice) => {
                        // try to convert our slice to a str
                        let converted = match std::str::from_utf8(fslice.as_slice()) {
                            Ok(converted) => converted,
                            Err(_) => {
                                return Err(scylla::deserialize::DeserializationError::new(
                                    crate::models::scylla_utils::errors::DeserializationError::ExpectedText,
                                ))
                            }
                        };
                        // get the correct kind
                        match #name::from_str(converted) {
                            Ok(event_type) => Ok(event_type),
                            Err(_) => Err(scylla::deserialize::DeserializationError::new(
                                crate::models::scylla_utils::errors::DeserializationError::UnknownValue,
                            )),
                        }
                    }
                    None => Err(scylla::deserialize::DeserializationError::new(
                        crate::models::scylla_utils::errors::DeserializationError::ExpectedNotNull,
                    )),
                }
            }
        }
    })
}

#[proc_macro_derive(ScyllaStoreAsStr)]
pub fn derive_scylla_store_as_str(stream: TokenStream) -> TokenStream {
    // parse our input struct
    let ast = syn::parse_macro_input!(stream as syn::DeriveInput);
    // get the name of our ident
    let name = &ast.ident;
    // start with an empty stream
    let mut output = quote! {};
    // add our json derives
    add_as_str_serialize(&mut output, name);
    add_as_str_deserialize(&mut output, name);
    output.into()
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
#[cfg(feature = "sync")]
#[proc_macro_attribute]
pub fn blocking_struct(args_raw: TokenStream, input: TokenStream) -> TokenStream {
    sync::blocking_struct(args_raw, input)
}

/// Create a wrapping, synchronous version of the given trait with the
/// suffix `Blocking` appended
///
/// The trait has a GAT that implements the target trait and a method that
/// returns an instance of the GAT from `&self`. Then all methods are just
/// called on that inner GAT. Any asynchronous methods are blocked on the
/// static runtime to ensure the trait is synchronous.
#[cfg(feature = "sync")]
#[proc_macro_attribute]
pub fn blocking_trait(meta: TokenStream, input: TokenStream) -> TokenStream {
    sync::blocking_trait(meta, input)
}

/// Create a blocking version of the async function
///
/// The body of the async function is just blocked on the static runtim
#[cfg(feature = "sync")]
#[proc_macro_attribute]
pub fn blocking_fn(meta: TokenStream, input: TokenStream) -> TokenStream {
    sync::blocking_fn(meta, input)
}

/// Set the struct to be a ``PyO3`` `pyclass`, applying `get` and `set` attributes to
/// public fields depending on the given args
#[cfg(feature = "python")]
#[proc_macro_attribute]
pub fn pyclass(meta: TokenStream, input: TokenStream) -> TokenStream {
    python::pyclass(meta, input)
}
