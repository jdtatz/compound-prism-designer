use darling::{ast, util, FromDeriveInput, ToTokens};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::hash::{Hash, Hasher};
use syn::parse::Parser;
use syn::{punctuated::Punctuated, DeriveInput, Ident, Path, Token, Type, WherePredicate};

#[derive(Debug, FromField)]
#[darling(attributes(wrapped_from))]
pub struct WrappedFromDeriveField {
    ident: Option<Ident>,
    ty: Type,
    #[darling(default)]
    skip: util::Flag,
}

fn unnamed_deconstructed_field(i: u32) -> Ident {
    format_ident!("_{}", i)
}

fn deconstruct_fields(fields: darling::ast::Fields<&WrappedFromDeriveField>) -> TokenStream {
    let mut i = 0;
    let ff = fields.map(|f| {
        let id = f
            .ident
            .as_ref()
            .cloned()
            .unwrap_or_else(|| unnamed_deconstructed_field(i));
        i += 1;
        id
    });
    quote!( #ff )
}

fn construct_fields(
    fields: darling::ast::Fields<&WrappedFromDeriveField>,
    impl_trait: &Path,
    function: &Ident,
) -> TokenStream {
    let mut i = 0;
    let ff = fields.map(|f| {
        let field = f
            .ident
            .as_ref()
            .cloned()
            .unwrap_or_else(|| unnamed_deconstructed_field(i));
        i += 1;
        let value = if f.skip.is_some() {
            quote!(#field)
        } else {
            quote!(#impl_trait :: #function ( #field ))
        };
        if let Some(ref id) = f.ident {
            quote!( #id : #value )
        } else {
            value
        }
    });
    quote!( #ff )
}

#[derive(Debug, FromVariant)]
#[darling(attributes(wrapped_from))]
pub struct WrappedFromDeriveVariant {
    ident: Ident,
    fields: darling::ast::Fields<WrappedFromDeriveField>,
    #[darling(default)]
    skip: util::Flag,
}

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(wrapped_from))]
pub struct WrappedFromDerive {
    ident: Ident,
    generics: syn::Generics,
    data: ast::Data<WrappedFromDeriveVariant, WrappedFromDeriveField>,
    #[darling(rename = "trait")]
    impl_trait: Path,
    function: Ident,
    #[darling(default)]
    bound: Option<String>,
}

struct RenameBoundAssociatedTypes<'s> {
    ident_map: &'s std::collections::HashMap<Ident, Ident>,
}

impl<'s> syn::visit_mut::VisitMut for RenameBoundAssociatedTypes<'s> {
    // FIXME much too generic need guards to prevent renaming of non-generics that share a name
    fn visit_ident_mut(&mut self, ident: &mut Ident) {
        if let Some(rpl) = self.ident_map.get(ident).cloned() {
            *ident = rpl;
        }
    }

    // fn visit_trait_bound_mut(&mut self, tbound: &mut TraitBound) {
    //     for s in tbound.path.segments.iter_mut() {
    //         // can ignore ident, b/c it will be a trait not a generic
    //         match &mut s.arguments {
    //             syn::PathArguments::None => {}
    //             syn::PathArguments::AngleBracketed(node) => {
    //                 syn::visit_mut::visit_angle_bracketed_generic_arguments_mut(self, node)
    //             }
    //             syn::PathArguments::Parenthesized(args) => {
    //             }
    //         }
    //     }
    // }
}

impl ToTokens for WrappedFromDerive {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let WrappedFromDerive {
            ref ident,
            ref generics,
            ref data,
            ref impl_trait,
            ref function,
            ref bound,
        } = *self;

        // Create unique generic type Ident mapping using hashes
        let wf_type_map: Vec<_> = generics
            .type_params()
            .map(|typ| {
                let id = typ.ident.clone();
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                ident.hash(&mut hasher);
                id.hash(&mut hasher);
                let hash = hasher.finish();
                let wf_ident = format_ident!("{}{:X}", id, hash);
                (id, wf_ident)
            })
            .collect();
        let wf_type_hashmap = wf_type_map.iter().cloned().collect();
        let mut fixer = RenameBoundAssociatedTypes {
            ident_map: &wf_type_hashmap,
        };

        // Generate ty generics for item
        let mut wf_generics = generics.clone();
        syn::visit_mut::visit_generics_mut(&mut fixer, &mut wf_generics);
        for typ in wf_generics.type_params_mut() {
            typ.default = None;
        }
        let (_, ty_generics, _) = generics.split_for_impl();
        let (_, wf_ty_generics, _) = wf_generics.split_for_impl();

        // create the combined generics for impl_generics & where_clause
        let mut combined_generics = generics.clone();
        for (i, typ) in wf_generics.type_params().enumerate() {
            combined_generics
                .params
                // FIXME syn only prints correct order for lifetimes and type params
                // not const params, so insert at start to ensure types come before const
                // and syn will automagically fix the lifetime-type order
                .insert(i, syn::GenericParam::Type(typ.clone()))
        }

        // Get or create Where-Clause
        let where_clause = combined_generics.make_where_clause();
        // let where_clause = where_clause.cloned().unwrap_or_else(|| WhereClause {
        //     predicates: Default::default(),
        //     where_token: Default::default(),
        // });

        // Append WrappedFrom bounds
        for (id, wf_ident) in wf_type_map.iter() {
            where_clause
                .predicates
                .push(syn::parse_quote!( #id : #impl_trait< #wf_ident > ));
        }
        // Append User-given bounds
        if let Some(mut bounds) = bound.clone() {
            for (id, wf_ident) in wf_type_hashmap.iter() {
                let from = format!("${}", id);
                let to = format!("{}", wf_ident);
                bounds = bounds.replace(from.as_str(), to.as_str());
            }
            let bounds: Punctuated<WherePredicate, Token![,]> = Punctuated::parse_terminated
                .parse_str(&bounds)
                .expect("FIXME(return compiler_error) Invalid bounds");
            for b in bounds.iter().cloned() {
                where_clause.predicates.push(b);
            }
        }

        let wf_arg = format_ident!("wf");
        let body = match data.as_ref() {
            ast::Data::Enum(variants) => {
                let arms = variants.into_iter().map(|v| {
                    let vid = &v.ident;
                    let vpath = quote!( #ident :: #vid );
                    let vfs = &v.fields.as_ref();
                    let decon = deconstruct_fields(vfs.clone());
                    if v.skip.is_some() {
                        quote! { #vpath #decon => #vpath #decon }
                    } else {
                        let wcon = construct_fields(vfs.clone(), &impl_trait, &function);
                        quote! { #vpath #decon => #vpath #wcon }
                    }
                });
                quote! {
                    match #wf_arg {
                        #( #arms ),*
                    }
                }
            }
            ast::Data::Struct(fields) => {
                let decon = deconstruct_fields(fields.clone());
                let wcon = construct_fields(fields.clone(), &impl_trait, &function);
                quote! {
                    let #ident #decon = #wf_arg ;
                    #ident #wcon
                }
            }
        };

        let (impl_generics, _, where_clause) = combined_generics.split_for_impl();

        tokens.extend(quote! {
            impl #impl_generics #impl_trait < #ident #wf_ty_generics > for #ident #ty_generics #where_clause {
                fn #function (#wf_arg: #ident #wf_ty_generics) -> Self {
                    #body
                }
            }
        });
    }
}

pub fn derive_helper_attr(input: &DeriveInput) -> Result<TokenStream, TokenStream> {
    WrappedFromDerive::from_derive_input(input)
        .map(|v| quote! { #v })
        .map_err(|e| e.write_errors())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test1() {
        let test_input = syn::parse_str(r###"
        #[derive(WrappedFrom)]
        #[wrapped_from(trait="crate::LossyFrom", function="lossy_from", bound="T::Item: LossyFrom<T_WF::Item>")]
        struct Test<T> {
            x: T,
            #[wrapped_from(skip)]
            y: f32,
            z: (),
        }
        "###).expect("Invalid test");
        let output = derive_helper_attr(&test_input);
        match output {
            Ok(tk) => println!("Test Succeded\n{}", tk),
            Err(tk) => panic!("Test Failed\n{}", tk),
        }
    }
}
