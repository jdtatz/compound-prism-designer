use crate::debug_assert_almost_eq;
use crate::glasscat::Glass;
use crate::utils::*;

#[derive(Debug, Display, Clone, Copy)]
pub enum RayTraceError {
    NoSurfaceIntersection,
    OutOfBounds,
    TotalInternalReflection,
    SpectrometerAngularResponseTooWeak,
}

impl Into<&'static str> for RayTraceError {
    fn into(self) -> &'static str {
        match self {
            RayTraceError::NoSurfaceIntersection => "NoSurfaceIntersection",
            RayTraceError::OutOfBounds => "OutOfBounds",
            RayTraceError::TotalInternalReflection => "TotalInternalReflection",
            RayTraceError::SpectrometerAngularResponseTooWeak => {
                "SpectrometerAngularResponseTooWeak"
            }
        }
    }
}

/// Collimated Polychromatic Gaussian Beam
#[derive(Debug, Clone)]
#[cfg_attr(not(target_arch = "nvptx64"), derive(Serialize, Deserialize))]
pub struct GaussianBeam<F: Float> {
    /// 1/e^2 beam width
    pub width: F,
    /// Mean y coordinate
    pub y_mean: F,
    /// Range of wavelengths
    pub w_range: (F, F),
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_arch = "nvptx64"), derive(Serialize, Deserialize))]
pub struct Surface<F: Float> {
    angle: F,
    normal: Pair<F>,
    midpt: Pair<F>,
}

impl<F: Float> Surface<F> {
    fn first_surface(angle: F, height: F) -> Self {
        let normal = rotate(
            angle,
            Pair {
                x: -F::one(),
                y: F::zero(),
            },
        );
        debug_assert_almost_eq!(
            (normal.y / normal.x).abs().to_f64(),
            angle.tan().abs().to_f64(),
            1e-10
        );
        Self {
            angle,
            normal,
            midpt: Pair {
                x: (normal.y / normal.x).abs() * height * F::from_f64(0.5),
                y: height * F::from_f64(0.5),
            },
        }
    }

    fn next_surface(&self, height: F, angle: F, sep_length: F) -> Self {
        let normal = rotate(
            angle,
            Pair {
                x: -F::one(),
                y: F::zero(),
            },
        );
        debug_assert_almost_eq!(
            (normal.y / normal.x).abs().to_f64(),
            angle.tan().abs().to_f64(),
            1e-10
        );
        let d1 = (self.normal.y / self.normal.x).abs() * height * F::from_f64(0.5);
        let d2 = (normal.y / normal.x).abs() * height * F::from_f64(0.5);
        let sep_dist = sep_length
            + if self.normal.y.is_sign_positive() != normal.y.is_sign_positive() {
                d1 + d2
            } else {
                (d1 - d2).abs()
            };
        Self {
            angle,
            normal,
            midpt: self.midpt
                + Pair {
                    x: sep_dist,
                    y: F::zero(),
                },
        }
    }

    pub fn end_points(&self, height: F) -> (Pair<F>, Pair<F>) {
        let dx = self.normal.y / self.normal.x * height * F::from_f64(0.5);
        let ux = self.midpt.x - dx;
        let lx = self.midpt.x + dx;
        (
            Pair { x: ux, y: height },
            Pair {
                x: lx,
                y: F::zero(),
            },
        )
    }
}

/// A Curved Surface, parameterized as a circular segment
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_arch = "nvptx64"), derive(Serialize, Deserialize))]
pub struct CurvedSurface<F: Float> {
    /// The midpt of the Curved Surface / circular segment
    midpt: Pair<F>,
    /// The center of the circle
    center: Pair<F>,
    /// The radius of the circle
    radius: F,
    /// max_dist_sq = sagitta ^ 2 + (chord_length / 2) ^ 2
    max_dist_sq: F,
}

impl<F: Float> CurvedSurface<F> {
    fn new(curvature: F, height: F, chord: Surface<F>) -> Self {
        let chord_length = height / chord.normal.x.abs();
        let radius = chord_length * F::from_f64(0.5) / curvature;
        let apothem = (radius * radius - chord_length * chord_length * F::from_f64(0.25)).sqrt();
        let sagitta = radius - apothem;
        let center = chord.midpt + chord.normal * apothem;
        let midpt = chord.midpt - chord.normal * sagitta;
        Self {
            midpt,
            center,
            radius,
            max_dist_sq: sagitta * sagitta + chord_length * chord_length * F::from_f64(0.25),
        }
    }

    fn is_along_arc(&self, pt: Pair<F>) -> bool {
        debug_assert!((pt - self.center).norm() < self.radius * F::from_f64(1.01));
        debug_assert!((pt - self.center).norm() > self.radius * F::from_f64(0.99));
        (pt - self.midpt).norm_squared() <= self.max_dist_sq
    }

    fn end_points(&self, height: F) -> (Pair<F>, Pair<F>) {
        let theta_2 =
            F::from_f64(2.) * (self.max_dist_sq.sqrt() / (F::from_f64(2.) * self.radius)).asin();
        let r = self.midpt - self.center;
        let u = self.center + rotate(theta_2, r);
        let l = self.center + rotate(-theta_2, r);
        debug_assert!(
            (u.y - height).abs() < F::from_f64(1e-4),
            "{:?} {}",
            u,
            height
        );
        debug_assert!(l.y.abs() < F::from_f64(1e-4), "{:?}", l);
        debug_assert!(
            ((u - r).norm_squared() - self.max_dist_sq).abs() / self.max_dist_sq
                < F::from_f64(1e-4)
        );
        debug_assert!(
            ((l - r).norm_squared() - self.max_dist_sq).abs() / self.max_dist_sq
                < F::from_f64(1e-4)
        );
        (
            Pair { x: u.x, y: height },
            Pair {
                x: l.x,
                y: F::zero(),
            },
        )
    }
}

/// Compound Prism Specification
#[derive(Debug, Clone)]
pub struct CompoundPrism<F: Float> {
    /// List of glasses the compound prism is composed of, in order.
    /// With their inter-media boundary surfaces
    prisms: arrayvec::ArrayVec<[(Glass, Surface<F>); 6]>,
    /// The curved lens-like last inter-media boundary surface of the compound prism
    lens: CurvedSurface<F>,
    /// Height of compound prism
    pub(crate) height: F,
    /// Width of compound prism
    pub(crate) width: F,
}

impl<F: Float> CompoundPrism<F> {
    /// Create a new Compound Prism Specification
    ///
    /// # Arguments
    ///  * `glasses` - List of glasses the compound prism is composed of, in order
    ///  * `angles` - Angles that parameterize the shape of the compound prism
    ///  * `lengths` - Lengths that parameterize the trapezoidal shape of the compound prism
    ///  * `curvature` - Lens Curvature of last surface of compound prism
    ///  * `height` - Height of compound prism
    ///  * `width` - Width of compound prism
    pub fn new<'a, I: IntoIterator<Item = &'a Glass>>(
        glasses: I,
        angles: &[F],
        lengths: &[F],
        curvature: F,
        height: F,
        width: F,
    ) -> Self
    where
        I::IntoIter: ExactSizeIterator,
    {
        let glasses = glasses.into_iter().cloned();
        debug_assert!(glasses.len() > 0);
        debug_assert!(angles.len() - 1 == glasses.len());
        debug_assert!(lengths.len() == glasses.len());
        let mut prisms = arrayvec::ArrayVec::new();
        let mut last_surface = Surface::first_surface(angles[0], height);
        for ((g, a), l) in glasses.zip(&angles[1..]).zip(lengths) {
            let next = last_surface.next_surface(height, *a, *l);
            prisms.push((g, last_surface));
            last_surface = next;
        }
        let lens = CurvedSurface::new(curvature, height, last_surface);
        Self {
            prisms,
            lens,
            height,
            width,
        }
    }

    #[cfg(not(target_arch = "nvptx64"))]
    pub fn polygons(&self) -> (Vec<[Pair<F>; 4]>, [Pair<F>; 4], Pair<F>, F) {
        let mut poly = Vec::with_capacity(self.prisms.len());
        let (mut u0, mut l0) = self.prisms[0].1.end_points(self.height);
        for (_, s) in self.prisms[1..].iter() {
            let (u1, l1) = s.end_points(self.height);
            poly.push([l0, u0, u1, l1]);
            u0 = u1;
            l0 = l1;
        }
        let (u1, l1) = self.lens.end_points(self.height);
        (poly, [l0, u0, u1, l1], self.lens.center, self.lens.radius)
    }
}

/// Array of detectors
#[derive(Debug, Clone)]
pub struct DetectorArray<'a, F: Float> {
    /// Boundaries of detection bins
    pub bins: &'a [[F; 2]],
    /// Minimum cosine of incident angle == cosine of maximum allowed incident angle
    min_ci: F,
    /// CCW angle of the array from normal = Rot(θ) @ (0, 1)
    angle: F,
    /// The normal of the array's surface, normal = Rot(θ) @ (-1, 0)
    normal: Pair<F>,
    /// Length of the array
    pub(crate) length: F,
}

impl<'a, F: Float> DetectorArray<'a, F> {
    pub fn new(bins: &'a [[F; 2]], min_ci: F, angle: F, length: F) -> Self {
        Self {
            bins,
            min_ci,
            angle,
            normal: rotate(
                angle,
                Pair {
                    x: -F::one(),
                    y: F::zero(),
                },
            ),
            length,
        }
    }

    pub fn end_points(&self, pos: &DetectorArrayPositioning<F>) -> (Pair<F>, Pair<F>) {
        (pos.position, pos.position + pos.direction * self.length)
    }
}

/// Positioning of detector array
#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy)]
#[cfg_attr(not(target_arch = "nvptx64"), derive(Serialize, Deserialize))]
pub struct DetectorArrayPositioning<F: Float> {
    /// Position vector of array
    pub position: Pair<F>,
    /// Unit direction vector of array
    pub direction: Pair<F>,
}

/// Light Ray
#[derive(Constructor, Debug, PartialEq, Clone, Copy)]
pub(crate) struct Ray<F: Float> {
    /// Origin position vector
    origin: Pair<F>,
    /// Unit normal direction vector
    direction: Pair<F>,
    /// S-Polarization Transmittance probability
    s_transmittance: F,
    /// P-Polarization Transmittance probability
    p_transmittance: F,
}

impl<F: Float> Ray<F> {
    /// Create a new unpolarized ray with full transmittance with a origin at (0, `y`) and a
    /// direction of (1, 0)
    ///
    /// # Arguments
    ///  * `y` - the initial y value of the ray's position
    pub fn new_from_start(y: F) -> Self {
        Ray {
            origin: Pair { x: F::zero(), y },
            direction: Pair {
                x: F::one(),
                y: F::zero(),
            },
            s_transmittance: F::one(),
            p_transmittance: F::one(),
        }
    }

    /// The average of the S & P Polarizations transmittance's
    fn average_transmittance(self) -> F {
        (self.s_transmittance + self.p_transmittance) * F::from_f64(0.5)
    }

    /// Refract ray through interface of two different media
    /// using vector form of snell's law
    ///
    /// # Arguments
    ///  * `intersection` - point of intersection between the media
    ///  * `normal` - the unit normal vector of the interface
    ///  * `ci` - cosine of incident angle
    ///  * `n1` - index of refraction of the current media
    ///  * `n2` - index of refraction of the new media
    fn refract(
        self,
        intersection: Pair<F>,
        normal: Pair<F>,
        ci: F,
        n1: F,
        n2: F,
    ) -> Result<Self, RayTraceError> {
        debug_assert!(n1 >= F::one());
        debug_assert!(n2 >= F::one());
        debug_assert!(normal.is_unit());
        let r = n1 / n2;
        let cr_sq = F::one() - r * r * (F::one() - ci * ci);
        if cr_sq < F::zero() {
            return Err(RayTraceError::TotalInternalReflection);
        }
        let cr = cr_sq.sqrt();
        let v = self.direction * r + normal * (r * ci - cr);
        let fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr);
        let fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci);
        Ok(Self {
            origin: intersection,
            direction: v,
            s_transmittance: self.s_transmittance * (F::one() - fresnel_rs * fresnel_rs),
            p_transmittance: self.p_transmittance * (F::one() - fresnel_rp * fresnel_rp),
        })
    }

    /// Find the intersection point of the ray with the interface
    /// of the current media and the next media. Using the line-plane intersection formula.
    /// Then refract the ray through the interface
    ///
    /// # Arguments
    ///  * `plane` - the inter-media interface plane
    ///  * `n1` - index of refraction of the current media
    ///  * `n2` - index of refraction of the new media
    ///  * `prism_height` - the height of the prism
    fn intersect_plane_interface(
        self,
        plane: &Surface<F>,
        n1: F,
        n2: F,
        prism_height: F,
    ) -> Result<Self, RayTraceError> {
        let ci = -self.direction.dot(plane.normal);
        if ci <= F::zero() {
            return Err(RayTraceError::OutOfBounds);
        }
        let d = (self.origin - plane.midpt).dot(plane.normal) / ci;
        let p = self.origin + self.direction * d;
        if p.y <= F::zero() || prism_height <= p.y {
            return Err(RayTraceError::OutOfBounds);
        }
        self.refract(p, plane.normal, ci, n1, n2)
    }

    /// Find the intersection point of the ray with the lens-like interface
    /// of current media and the next media. Using the line-sphere intersection formula.
    /// Then refract the ray through the interface
    ///
    /// # Arguments
    ///  * `lens` - the parameterized curved surface of the lens
    ///  * `n1` - index of refraction of the current media
    ///  * `n2` - index of refraction of the new media
    fn intersect_curved_interface(
        self,
        lens: &CurvedSurface<F>,
        n1: F,
        n2: F,
    ) -> Result<Self, RayTraceError> {
        let delta = self.origin - lens.center;
        let ud = self.direction.dot(delta);
        let under = ud * ud - delta.norm_squared() + lens.radius * lens.radius;
        if under < F::zero() {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let d = -ud + under.sqrt();
        if d <= F::zero() {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let p = self.origin + self.direction * d;
        if !lens.is_along_arc(p) {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let snorm = (lens.center - p) / lens.radius;
        debug_assert!(snorm.is_unit());
        self.refract(p, snorm, -self.direction.dot(snorm), n1, n2)
    }

    /// Find the intersection position of the ray with the detector array
    /// and the ray's transmission probability. The intersection position is a
    /// scalar on the line defined by the detector array.
    ///
    /// # Arguments
    ///  * `detarr` - detector array specification
    ///  * `detpos` - the position and orientation of the detector array
    fn intersect_detector_array(
        self,
        detarr: &DetectorArray<F>,
        detpos: &DetectorArrayPositioning<F>,
    ) -> Result<(Pair<F>, F, F), RayTraceError> {
        let ci = -self.direction.dot(detarr.normal);
        if ci <= detarr.min_ci {
            return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
        }
        let d = (self.origin - detpos.position).dot(detarr.normal) / ci;
        debug_assert!(d > F::zero());
        let p = self.origin + self.direction * d;
        debug_assert!((detpos.direction).is_unit());
        let pos = (p - detpos.position).dot(detpos.direction);
        if pos < F::zero() || detarr.length < pos {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        Ok((p, pos, self.average_transmittance()))
    }

    /// Propagate a ray of `wavelength` through the compound prism
    ///
    /// # Arguments
    ///  * `cmpnd` - the compound prism specification
    ///  * `wavelength` - the wavelength of the light ray
    fn propagate_internal(
        self,
        cmpnd: &CompoundPrism<F>,
        wavelength: F,
    ) -> Result<Self, RayTraceError> {
        let (ray, n1) =
            cmpnd
                .prisms
                .iter()
                .try_fold((self, F::one()), |(ray, n1), (glass, plane)| {
                    let n2 = F::from_f64(glass.calc_n(wavelength.to_f64()));
                    debug_assert!(n2 >= F::one());
                    let ray = ray.intersect_plane_interface(plane, n1, n2, cmpnd.height)?;
                    Ok((ray, n2))
                })?;
        let n2 = F::one();
        ray.intersect_curved_interface(&cmpnd.lens, n1, n2)
    }

    /// Propagate a ray of `wavelength` through the compound prism and
    /// intersect the detector array. Returning the intersection scalar
    /// and the transmission probability.
    ///
    /// # Arguments
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `cmpnd` - the compound prism specification
    ///  * `detarr` - detector array specification
    ///  * `detpos` - the position and orientation of the detector array
    pub fn propagate(
        self,
        wavelength: F,
        cmpnd: &CompoundPrism<F>,
        detarr: &DetectorArray<F>,
        detpos: &DetectorArrayPositioning<F>,
    ) -> Result<(Pair<F>, F, F), RayTraceError> {
        self.propagate_internal(cmpnd, wavelength)?
            .intersect_detector_array(detarr, detpos)
    }

    /// Propagate a ray of `wavelength` through the compound prism and
    /// intersect the detector array. Returning an iterator of the ray's origin position and
    /// all of the intersection positions.
    ///
    /// # Arguments
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `cmpnd` - the compound prism specification
    ///  * `detarr` - detector array specification
    ///  * `detpos` - the position and orientation of the detector array
    fn trace<'s>(
        self,
        wavelength: F,
        cmpnd: &'s CompoundPrism<F>,
        detarr: &'s DetectorArray<'s, F>,
        detpos: &'s DetectorArrayPositioning<F>,
    ) -> impl Iterator<Item = Result<Pair<F>, RayTraceError>> + 's {
        let mut ray = self;
        let mut n1 = F::one();
        let mut prisms = cmpnd.prisms.iter().fuse();
        let mut internal = true;
        let mut done = false;
        let mut propagation_fn = move || -> Result<Option<Pair<F>>, RayTraceError> {
            match prisms.next() {
                Some((glass, plane)) => {
                    let n2 = F::from_f64(glass.calc_n(wavelength.to_f64()));
                    ray = ray.intersect_plane_interface(plane, n1, n2, cmpnd.height)?;
                    n1 = n2;
                    Ok(Some(ray.origin))
                }
                None if !done && internal => {
                    internal = false;
                    let n2 = F::one();
                    ray = ray.intersect_curved_interface(&cmpnd.lens, n1, n2)?;
                    Ok(Some(ray.origin))
                }
                None if !done && !internal => {
                    done = true;
                    let (pos, _, _) = ray.intersect_detector_array(detarr, detpos)?;
                    Ok(Some(pos))
                }
                _ if done => Ok(None),
                _ => unreachable!(),
            }
        };
        core::iter::once(Ok(self.origin))
            .chain(core::iter::from_fn(move || propagation_fn().transpose()).fuse())
    }
}

/// Find the position and orientation of the detector array,
/// parameterized by the minimum and maximum wavelengths of the input beam,
/// and its angle from the normal.
///
/// # Arguments
///  * `cmpnd` - the compound prism specification
///  * `detarr` - detector array specification
///  * `beam` - input gaussian beam specification
pub fn detector_array_positioning<F: Float>(
    cmpnd: &CompoundPrism<F>,
    detarr: &DetectorArray<F>,
    beam: &GaussianBeam<F>,
) -> Result<DetectorArrayPositioning<F>, RayTraceError> {
    let ray = Ray::new_from_start(beam.y_mean);
    let (wmin, wmax) = beam.w_range;
    let lower_ray = ray.propagate_internal(cmpnd, wmin)?;
    let upper_ray = ray.propagate_internal(cmpnd, wmax)?;
    if lower_ray.average_transmittance() <= F::from_f64(1e-3)
        || upper_ray.average_transmittance() <= F::from_f64(1e-3)
    {
        return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
    }
    debug_assert!(lower_ray.direction.is_unit());
    debug_assert!(upper_ray.direction.is_unit());
    let spec_dir = rotate(
        detarr.angle,
        Pair {
            x: F::zero(),
            y: F::one(),
        },
    );
    let spec = spec_dir * detarr.length;
    let mat = Mat2::new_from_cols(upper_ray.direction, -lower_ray.direction);
    let imat = mat.inverse().ok_or(RayTraceError::NoSurfaceIntersection)?;
    let dists = imat * (spec - upper_ray.origin + lower_ray.origin);
    let d2 = dists.y;
    let l_vertex = lower_ray.origin + lower_ray.direction * d2;
    let (pos, dir) = if d2 > F::zero() {
        (l_vertex, spec_dir)
    } else {
        let dists = imat * (-spec - upper_ray.origin + lower_ray.origin);
        let d2 = dists.y;
        if d2 < F::zero() {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let u_vertex = lower_ray.origin + lower_ray.direction * d2;
        (u_vertex, -spec_dir)
    };
    Ok(DetectorArrayPositioning {
        position: pos,
        direction: dir,
    })
}

/// Trace the propagation of a ray of `wavelength` through the compound prism and
/// intersection the detector array. Returning an iterator of the ray's origin position and
/// all of the intersection positions.
///
/// # Arguments
///  * `wavelength` - the wavelength of the light ray
///  * `init_y` - the inital y value of the ray
///  * `cmpnd` - the compound prism specification
///  * `detarr` - detector array specification
///  * `detpos` - the position and orientation of the detector array
pub fn trace<'s, F: Float>(
    wavelength: F,
    init_y: F,
    cmpnd: &'s CompoundPrism<F>,
    detarr: &'s DetectorArray<'s, F>,
    detpos: &'s DetectorArrayPositioning<F>,
) -> impl Iterator<Item = Result<Pair<F>, RayTraceError>> + 's {
    let ray = Ray::new_from_start(init_y);
    ray.trace(wavelength, cmpnd, detarr, detpos)
}

pub struct Spectrometer<'a, F: Float> {
    pub gaussian_beam: GaussianBeam<F>,
    pub compound_prism: CompoundPrism<F>,
    pub detector_array: DetectorArray<'a, F>,
    pub detector_array_position: DetectorArrayPositioning<F>,
}

impl<'a, F: Float> Spectrometer<'a, F> {
    pub fn new(
        gaussian_beam: GaussianBeam<F>,
        compound_prism: CompoundPrism<F>,
        detector_array: DetectorArray<'a, F>,
    ) -> Result<Self, RayTraceError> {
        let detector_array_position =
            detector_array_positioning(&compound_prism, &detector_array, &gaussian_beam)?;
        Ok(Self {
            gaussian_beam,
            compound_prism,
            detector_array,
            detector_array_position,
        })
    }

    pub fn propagate(&self, wavelength: F, initial_y: F) -> Result<(F, F), RayTraceError> {
        Ray::new_from_start(initial_y)
            .propagate(
                wavelength,
                &self.compound_prism,
                &self.detector_array,
                &self.detector_array_position,
            )
            .map(|(_p, pos, t)| (pos, t))
    }

    pub fn trace_ray_path<'s>(
        &'s self,
        wavelength: F,
        initial_y: F,
    ) -> impl Iterator<Item = Result<Pair<F>, RayTraceError>> + 's {
        Ray::new_from_start(initial_y).trace(
            wavelength,
            &self.compound_prism,
            &self.detector_array,
            &self.detector_array_position,
        )
    }
}
