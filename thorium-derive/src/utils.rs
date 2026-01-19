//! Helpful functions to share between modules

use std::collections::HashMap;

use syn::{GenericArgument, Ident, PathArguments, Type, TypePath};

/// Recursively walk a mutable `Type`, replacing the first identifier that
/// appears in `type_map`. Returns a reference to the new identifier as
/// soon as a replacement occurs.
pub fn replace_type<'a>(ty: &mut Type, type_map: &'a HashMap<Ident, Ident>) -> Option<&'a Ident> {
    match ty {
        Type::Path(TypePath { qself: None, path }) => {
            // try to replace the *base* identifier of any segment; leave any
            // generic arguments untouched
            for seg in &mut path.segments {
                if let Some(new_id) = type_map.get(&seg.ident) {
                    seg.ident = new_id.clone();
                    return Some(new_id);
                }
            }
            // recurse into generic arguments
            if let Some(last_seg) = path.segments.last_mut()
                && let PathArguments::AngleBracketed(ref mut args) = last_seg.arguments
            {
                for generic in &mut args.args {
                    if let GenericArgument::Type(inner_ty) = generic
                        && let Some(repl) = replace_type(inner_ty, type_map)
                    {
                        return Some(repl);
                    }
                }
            }
            None
        }
        Type::Reference(r) => replace_type(&mut r.elem, type_map),
        Type::Ptr(p) => replace_type(&mut p.elem, type_map),
        Type::Slice(s) => replace_type(&mut s.elem, type_map),
        Type::Array(a) => replace_type(&mut a.elem, type_map),
        Type::Tuple(tup) => {
            for elem in &mut tup.elems {
                if let Some(repl) = replace_type(elem, type_map) {
                    return Some(repl);
                }
            }
            None
        }
        Type::Paren(p) => replace_type(&mut p.elem, type_map),
        Type::Group(g) => replace_type(&mut g.elem, type_map),
        _ => None,
    }
}
