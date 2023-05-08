use proc_macro::TokenStream;

#[macro_use]
extern crate darling;

mod core;
use darling::FromDeriveInput;
use quote::ToTokens;
use syn::parse_macro_input;

use crate::core::{WrappedFromDerive, WrappedFromTupleImplFn};

#[proc_macro]
pub fn wrapped_from_tuples(input: TokenStream) -> TokenStream {
    TokenStream::from(parse_macro_input!(input as WrappedFromTupleImplFn).into_token_stream())
}

#[proc_macro_derive(WrappedFrom, attributes(wrapped_from))]
pub fn derive_helper_attr(input: TokenStream) -> TokenStream {
    TokenStream::from(
        WrappedFromDerive::from_derive_input(&parse_macro_input!(input))
            .map_or_else(|err| err.write_errors(), ToTokens::into_token_stream),
    )
}
