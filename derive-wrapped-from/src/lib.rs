use proc_macro::TokenStream;

#[macro_use]
extern crate darling;

mod core;
use syn::parse_macro_input;

#[proc_macro_derive(WrappedFrom, attributes(wrapped_from))]
pub fn derive_helper_attr(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input);
    TokenStream::from(crate::core::derive_helper_attr(&ast).map_or_else(|err| err, |ok| ok))
}
