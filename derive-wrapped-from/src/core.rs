use darling::{ast, util, FromDeriveInput, ToTokens};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use std::hash::{Hash, Hasher};
use std::{collections::HashMap, ops::Range};
use syn::{fold::Fold, spanned::Spanned, TypeParam, TypePath, WherePredicate};
use syn::{punctuated::Punctuated, Ident, LitStr, Path, Token, Type};

fn associate_bounds(
    typ: &syn::Type,
    ident_map: &HashMap<Ident, Ident>,
) -> Vec<(TypePath, TypePath)> {
    match typ {
        Type::Path(tp) => {
            let mut fixer = RenameImpliedBoundTypes {
                ident_map,
                fixed: false,
            };
            let other = fixer.fold_type_path(tp.clone());
            if tp != &other {
                vec![(tp.clone(), other)]
            } else {
                Vec::new()
            }
        }
        // Never type implments everything automatically
        Type::Never(_) => Vec::new(),
        // Container variants
        Type::Array(syn::TypeArray { elem, .. }) => associate_bounds(&elem, ident_map),
        Type::Slice(syn::TypeSlice { elem, .. }) => associate_bounds(&elem, ident_map),
        Type::Tuple(syn::TypeTuple { elems, .. }) => elems
            .iter()
            .flat_map(|elem| associate_bounds(elem, ident_map))
            .collect(),
        Type::Group(syn::TypeGroup { elem, .. }) | Type::Paren(syn::TypeParen { elem, .. }) => {
            associate_bounds(&elem, ident_map)
        }
        #[cfg(feature = "allow-reference-types")]
        Type::Reference(syn::TypeReference { elem, .. }) | Type::Ptr(syn::TypePtr { elem, .. }) => {
            associate_bounds(&elem, ident_map)
        }
        // Unpported/impossible variants
        #[cfg(not(feature = "allow-reference-types"))]
        Type::Ptr(_) | Type::Reference(_) => {
            panic!("From-like traits should only be implmented for types that own their data")
        }
        Type::ImplTrait(_) => panic!("impl Trait is not allowed for struct/enum fields"),
        Type::Infer(_) => panic!("Infering types is not supported for struct/enum fields"),
        Type::Macro(_) => panic!("Macros in type position should be resolved before derive macros"),
        // Unimplemented variants
        Type::BareFn(_) => unimplemented!("BareFn: {:?} is not yet supported", typ),
        Type::TraitObject(_) => unimplemented!("TraitObject: {:?} is not yet supported", typ),
        Type::Verbatim(v) => panic!("Unparseable tokens: {:?}", v),
        #[cfg_attr(test, deny(non_exhaustive_omitted_patterns))]
        _ => unimplemented!("type: {:?} is not yet supported", typ),
    }
}

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

fn deconstructed_field(field: &WrappedFromDeriveField, i: u32) -> Ident {
    field
        .ident
        .as_ref()
        .cloned()
        .unwrap_or_else(|| unnamed_deconstructed_field(i))
}

fn deconstruct_fields(fields: darling::ast::Fields<&WrappedFromDeriveField>) -> TokenStream {
    let mut i = 0;
    let ff = fields.map(|f| {
        let id = deconstructed_field(f, i);
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
        let field = deconstructed_field(f, i);
        i += 1;
        if f.skip.is_present() {
            quote!(#field)
        } else {
            let value = quote!(#impl_trait :: #function ( #field ));
            if let Some(ref id) = f.ident {
                quote!( #id : #value )
            } else {
                value
            }
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
    bound: Option<darling::util::SpannedValue<String>>,
}

// Only needed till RFC 2089: Implied bounds, is stabilized
struct RenameImpliedBoundTypes<'s> {
    ident_map: &'s HashMap<Ident, Ident>,
    fixed: bool,
}

impl<'s> RenameImpliedBoundTypes<'s> {
    fn maybe_fix_ident(&mut self, ident: &mut Ident) {
        if let Some(rpl) = self.ident_map.get(ident).cloned() {
            self.fixed = true;
            *ident = rpl;
        }
    }
}

impl<'s> syn::fold::Fold for RenameImpliedBoundTypes<'s> {
    fn fold_type_param(&mut self, mut type_param: TypeParam) -> TypeParam {
        self.maybe_fix_ident(&mut type_param.ident);
        syn::fold::fold_type_param(self, type_param)
    }

    fn fold_type_path(&mut self, tp: TypePath) -> TypePath {
        let TypePath {
            mut qself,
            mut path,
        } = tp;
        // Generic param can only be in QSelf xor the 1st path ident w/o arguments, and/or in the path segment arguments
        if let Some(q) = qself {
            qself = Some(syn::fold::fold_qself(self, q))
        } else if let Some(syn::PathSegment {
            ident,
            arguments: syn::PathArguments::None,
        }) = path.segments.first_mut()
        {
            self.maybe_fix_ident(ident)
        }
        let path = syn::fold::fold_path(self, path);
        TypePath { qself, path }
    }
}

struct EnsureSizedTypeParams;

impl syn::fold::Fold for EnsureSizedTypeParams {
    fn fold_trait_bound(&mut self, bound: syn::TraitBound) -> syn::TraitBound {
        if matches!(bound.modifier, syn::TraitBoundModifier::Maybe(_))
            && bound.path == Ident::new("Sized", Span::call_site()).into()
        {
            syn::TraitBound {
                modifier: syn::TraitBoundModifier::None,
                ..bound
            }
        } else {
            bound
        }
    }
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

        let generics = syn::fold::fold_generics(&mut EnsureSizedTypeParams, generics.clone());

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
        let mut fixer = RenameImpliedBoundTypes {
            ident_map: &wf_type_hashmap,
            fixed: false,
        };

        let assoc = match data.as_ref() {
            ast::Data::Enum(vs) => vs
                .into_iter()
                .filter(|v| !v.skip.is_present())
                .flat_map(|v| v.fields.iter())
                .filter(|f| !f.skip.is_present())
                .flat_map(|f| associate_bounds(&f.ty, &wf_type_hashmap))
                .collect(),
            ast::Data::Struct(fs) => fs
                .into_iter()
                .filter(|f| !f.skip.is_present())
                .flat_map(|f| associate_bounds(&f.ty, &wf_type_hashmap))
                .collect::<Vec<_>>(),
        };
        #[cfg(test)]
        for (to, from) in assoc.iter() {
            println!("add bound, `{}` from `{}`", quote!(#to), quote!(#from));
        }
        let assoc = assoc.into_iter().collect::<HashMap<_, _>>();

        // Generate ty generics for item
        let wf_generics = syn::fold::fold_generics(&mut fixer, generics.clone());
        assert!(fixer.fixed);
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

        // Append WrappedFrom bounds
        for (id, wf_ident) in wf_type_map.iter() {
            where_clause
                .predicates
                .push(syn::parse_quote!( #id : #impl_trait< #wf_ident > ));
        }
        // Append Genearted WrappedFrom bounds for Associated Types
        for (a_ty, wf_a_ty) in assoc {
            where_clause
                .predicates
                .push(syn::parse_quote!( #a_ty : #impl_trait< #wf_a_ty > ));
        }
        // Append User-given bounds
        if let Some(mut bounds) = bound.clone() {
            for (id, wf_ident) in wf_type_hashmap.iter() {
                let from = format!("${}", id);
                let to = format!("{}", wf_ident);
                *bounds = bounds.replace(from.as_str(), to.as_str());
            }
            let bounds = LitStr::new(&*bounds, bounds.span());
            let bounds: Punctuated<WherePredicate, Token![,]> =
                match bounds.parse_with(Punctuated::parse_terminated) {
                    Ok(p) => p,
                    Err(e) => {
                        return tokens.extend(e.into_compile_error());
                    }
                };
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
                    if v.skip.is_present() {
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

fn resolve_expr_range(expr_range: syn::ExprRange) -> syn::Result<Range<usize>> {
    let syn::ExprRange {
        from, limits, to, ..
    } = expr_range;
    let lower = match from.as_deref() {
        Some(syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(i),
            ..
        })) => i.base10_parse()?,
        None => 0usize,
        _ => {
            return Err(syn::Error::new(
                from.span(),
                "lower bound is a non-integer literal",
            ));
        }
    };
    let upper = if let Some(syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(i),
        ..
    })) = to.as_deref()
    {
        let u = i.base10_parse()?;
        match limits {
            syn::RangeLimits::HalfOpen(_) => u,
            syn::RangeLimits::Closed(_) => u + 1,
        }
    } else {
        return Err(syn::Error::new(
            to.span(),
            "upper bound is not an integer literal",
        ));
    };
    Ok(lower..upper)
}

pub struct WrappedFromTupleImplFn {
    impl_trait: Path,
    function: Ident,
    range: Range<usize>,
}

impl syn::parse::Parse for WrappedFromTupleImplFn {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut path = input.call(Path::parse_mod_style)?;
        let syn::PathSegment {
            ident: function,
            arguments,
        } = path
            .segments
            .pop()
            .map(syn::punctuated::Pair::into_value)
            .ok_or_else(|| syn::Error::new(path.span(), "Empty trait::function path"))?;
        assert!(
            arguments.is_empty(),
            "This shouldn't be raised, the path was parsed mod-style"
        );
        let impl_trait = path;
        input.parse::<Token![for]>()?;
        let range = resolve_expr_range(input.parse()?)?;
        Ok(WrappedFromTupleImplFn {
            impl_trait,
            function,
            range,
        })
    }
}

pub fn impl_for_tuple(impl_trait: &Path, function: &Ident, n: usize) -> TokenStream {
    let idx = (0..n).map(syn::Index::from);
    let from = (0..n).map(|i| format_ident!("T{}", i)).collect::<Vec<_>>();
    let to = (0..n).map(|i| format_ident!("U{}", i)).collect::<Vec<_>>();
    let mut call = impl_trait.clone();
    call.segments.push(function.clone().into());
    quote! {
        impl < #( #from, #to: #impl_trait < #from > ),* > #impl_trait < ( #( #from, )* ) > for ( #( #to, )* ) {
            fn #function (from: ( #( #from, )* )) -> Self {
                (
                    #(
                        #call ( from . #idx ),
                    )*
                )
            }
        }
    }
}

impl ToTokens for WrappedFromTupleImplFn {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(
            self.range
                .clone()
                .flat_map(|i| impl_for_tuple(&self.impl_trait, &self.function, i)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::process::{Command, Stdio};

    fn format_tokens(tokens: TokenStream) -> String {
        let tks = tokens.to_string();
        let mut child = Command::new("rustfmt")
            .arg("--emit")
            .arg("stdout")
            .arg("--edition")
            .arg("2018")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("failed to spawn process");
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(tks.as_bytes()).unwrap();
        stdin.flush().unwrap();
        drop(stdin);

        let out = child.wait_with_output().unwrap();
        String::from_utf8_lossy(&out.stdout).to_string()
    }

    #[test]
    fn test1() {
        let test_input = syn::parse_str(r###"
        #[derive(WrappedFrom)]
        #[wrapped_from(trait="crate::LossyFrom", function="lossy_from", bound="T::Item: LossyFrom<$T::Item>")]
        struct Test<T: Copy> {
            x: T,
            #[wrapped_from(skip)]
            y: f32,
            z: T::Scalar,
            a: <Vec<T> as idk::what<T::Scalar>::to>::Assoc
        }
        "###).expect("Invalid test");
        let output = WrappedFromDerive::from_derive_input(&test_input);
        match output {
            Ok(tk) => println!("Test Succeded\n{}", format_tokens(tk.into_token_stream())),
            Err(tk) => panic!("Test Failed\n{}", tk),
        }
    }

    #[test]
    fn test2() {
        let test_input: WrappedFromTupleImplFn =
            syn::parse_str(r##"LossyFrom::lossy_from for ..=12"##).expect("Test Failed");
        let output = test_input.into_token_stream();
        println!("Test Succeded\n{}", format_tokens(output));
    }
}
