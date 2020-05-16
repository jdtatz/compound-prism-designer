use crate::glasscat::Glass;
use crate::utils::*;
use crate::geom::*;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianBeam<F: Float> {
    /// 1/e^2 beam width
    pub width: F,
    /// Mean y coordinate
    pub y_mean: F,
    /// Range of wavelengths
    pub w_range: (F, F),
}

impl<F: Float> GaussianBeam<F> {
    pub fn inverse_cdf_wavelength(&self, p: F) -> F {
        self.w_range.0 + (self.w_range.1 - self.w_range.0) * p
    }

    pub fn inverse_cdf_initial_y(&self, p: F) -> F {
        use core::f64::consts::FRAC_1_SQRT_2;
        self.y_mean
            - self.width * F::from_f64(FRAC_1_SQRT_2) * crate::erf::erfc_inv(F::from_f64(2.) * p)
    }

    pub fn inverse_cdf_ray(&self, p: F) -> Ray<F> {
        Ray::new_from_start(self.inverse_cdf_initial_y(p))
    }
}

/// Polychromatic Uniform Circular Multi-Mode Fiber Beam
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiberBeam<F: Float> {
    /// Radius of fiber core
    pub radius: F,
    /// Numerical apeature
    pub na: F,
    /// Mean y coordinate
    pub y_mean: F,
    /// Range of wavelengths
    pub w_range: (F, F),
}

/// Compound Prism Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundPrism<F: Float> {
    /// List of glasses the compound prism is composed of, in order.
    /// With their inter-media boundary surfaces
    prisms: arrayvec::ArrayVec<[(Glass<F>, Plane<F>); 6]>,
    /// The curved lens-like last inter-media boundary surface of the compound prism
    lens: CurvedPlane<F>,
    /// Height of compound prism
    pub(crate) height: F,
    /// Width of compound prism
    pub(crate) width: F,
    /// Are the inter-media surfaces coated(anti-reflective)?
    ar_coated: bool
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
    ///  * `coat` - Coat the outer compound prism surfaces with anti-reflective coating
    pub fn new<I: IntoIterator<Item = Glass<F>>>(
        glasses: I,
        angles: &[F],
        lengths: &[F],
        curvature: F,
        height: F,
        width: F,
        coat: bool,
    ) -> Self
    where
        I::IntoIter: ExactSizeIterator,
    {
        let glasses = glasses.into_iter();
        debug_assert!(glasses.len() > 0);
        debug_assert!(angles.len() - 1 == glasses.len());
        debug_assert!(lengths.len() == glasses.len());
        let mut prisms = arrayvec::ArrayVec::new();
        let (mut last_surface, rest) = create_joined_trapezoids(height, angles, lengths);
        for (g, next_surface) in glasses.zip(rest) {
            prisms.push((g, last_surface));
            last_surface = next_surface;
        }
        let lens = CurvedPlane::new(curvature, height, last_surface);
        Self {
            prisms,
            lens,
            height,
            width,
            ar_coated: coat
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

/// Linear Array of detectors
/// where the bins are defined by
/// for i in 0..bin_count
/// lower_bound = linear_slope * i + linear_intercept
/// upper_bound = linear_slope * i + linear_intercept + bin_size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearDetectorArray<F: Float> {
    /// The number of bins in the array
    pub(crate) bin_count: u32,
    /// The size / length of the bins
    bin_size: F,
    /// The slope used in the linear equation find the bin bounds
    linear_slope: F,
    /// The intercept used in the linear equation find the bin bounds
    linear_intercept: F,
    /// Minimum cosine of incident angle == cosine of maximum allowed incident angle
    min_ci: F,
    /// CCW angle of the array from normal = Rot(θ) @ (0, 1)
    angle: F,
    /// The normal of the array's surface, normal = Rot(θ) @ (-1, 0)
    normal: Pair<F>,
    /// Length of the array
    pub(crate) length: F,
}

impl<F: Float> LinearDetectorArray<F> {
    pub fn new(
        bin_count: u32,
        bin_size: F,
        linear_slope: F,
        linear_intercept: F,
        min_ci: F,
        angle: F,
        length: F,
    ) -> Self {
        debug_assert!(bin_count > 0);
        debug_assert!(bin_size > F::zero());
        debug_assert!(linear_slope > F::zero());
        debug_assert!(linear_intercept >= F::zero());
        Self {
            bin_count,
            bin_size,
            linear_slope,
            linear_intercept,
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

    pub fn bin_index(&self, pos: F) -> Option<u32> {
        let (bin, bin_pos) = (pos - self.linear_intercept).euclid_dev_rem(self.linear_slope);
        let bin = bin.to_u32();
        if bin < self.bin_count && bin_pos < self.bin_size {
            Some(bin)
        } else {
            None
        }
    }

    pub fn bounds<'s>(&'s self) -> impl ExactSizeIterator<Item = [F; 2]> + 's {
        (0..self.bin_count).map(move |i| {
            let i = F::from_f64(i as f64);
            let lb = self.linear_intercept + self.linear_slope * i;
            let ub = lb + self.bin_size;
            [lb, ub]
        })
    }

    #[cfg(not(target_arch = "nvptx64"))]
    pub fn end_points(&self, pos: &DetectorArrayPositioning<F>) -> (Pair<F>, Pair<F>) {
        (pos.position, pos.position + pos.direction * self.length)
    }
}

/// Positioning of detector array
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct DetectorArrayPositioning<F: Float> {
    /// Position vector of array
    pub position: Pair<F>,
    /// Unit direction vector of array
    pub direction: Pair<F>,
}

/// Light Ray
#[derive(Constructor, Debug, PartialEq, Clone, Copy)]
pub struct Ray<F: Float> {
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

    /// Create a new unpolarized ray with full transmittance with a origin at (0, `y`) and a
    /// direction of (angle_cosine, angle_sine)
    ///
    /// # Arguments
    ///  * `y` - the initial y value of the ray's position
    ///  * `angle_sine` - the sine of the angle that the ray is rotated from the axis
    pub fn new_from_start_at_angle(y: F, angle_sine: F) -> Self {
        Ray {
            origin: Pair { x: F::zero(), y },
            direction: Pair {
                x: (F::one() - angle_sine.sqr()).sqrt(),
                y: angle_sine,
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
        n1: F,
        n2: F,
        ar_coated: bool,
    ) -> Result<Self, RayTraceError> {
        debug_assert!(n1 >= F::one());
        debug_assert!(n2 >= F::one());
        debug_assert!(normal.is_unit());
        let ci = -self.direction.dot(normal);
        let r = n1 / n2;
        let cr_sq = F::one() - r.sqr() * (F::one() - ci.sqr());
        if cr_sq < F::zero() {
            return Err(RayTraceError::TotalInternalReflection);
        }
        let cr = cr_sq.sqrt();
        let v = self.direction * r + normal * (r * ci - cr);
        let (s_transmittance, p_transmittance) = if ar_coated && ci > F::from_f64(0.5) {
            (self.s_transmittance * F::from_f64(0.99), self.p_transmittance * F::from_f64(0.99))
        } else {
            let fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr);
            let fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci);
            (self.s_transmittance * (F::one() - fresnel_rs.sqr()),
             self.p_transmittance * (F::one() - fresnel_rp.sqr()))
        };
        Ok(Self {
            origin: intersection,
            direction: v,
            s_transmittance,
            p_transmittance,
        })
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
        detarr: &LinearDetectorArray<F>,
        detpos: &DetectorArrayPositioning<F>,
    ) -> Result<(Pair<F>, F, F), RayTraceError> {
        let ci = -self.direction.dot(detarr.normal);
        if ci <= detarr.min_ci {
            return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
        }
        let d = (self.origin - detpos.position).dot(detarr.normal) / ci;
        debug_assert!(d > F::zero());
        let p = self.direction.mul_add(d, self.origin);
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
        dbg!(cmpnd);
        let (ray, n1) =
            cmpnd
                .prisms
                .iter()
                .try_fold((self, F::one()), |(ray, n1), (glass, plane)| {
                    let n2 = glass.calc_n(wavelength);
                    debug_assert!(n2 >= F::one());
                    dbg!(ray);
                    let (p, normal) = plane.intersection((ray.origin, ray.direction)).ok_or(RayTraceError::NoSurfaceIntersection)?;
                    let ray = self.refract(p, normal, n1, n2, cmpnd.ar_coated)?;
                    Ok((ray, n2))
                })?;
        let n2 = F::one();
        let (p, normal) = cmpnd.lens.intersection((ray.origin, ray.direction)).ok_or(RayTraceError::NoSurfaceIntersection)?;
        dbg!(ray);
        self.refract(p, normal, n1, n2, cmpnd.ar_coated)
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
        detarr: &LinearDetectorArray<F>,
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
    #[cfg(not(target_arch = "nvptx64"))]
    fn trace<'s>(
        self,
        wavelength: F,
        cmpnd: &'s CompoundPrism<F>,
        detarr: &'s LinearDetectorArray<F>,
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
                    let n2 = glass.calc_n(wavelength);
                    let (p, normal) = plane.intersection((ray.origin, ray.direction)).ok_or(RayTraceError::NoSurfaceIntersection)?;
                    ray = self.refract(p, normal, n1, n2, cmpnd.ar_coated)?;
                    n1 = n2;
                    Ok(Some(ray.origin))
                }
                None if !done && internal => {
                    internal = false;
                    let n2 = F::one();
                    let (p, normal) = cmpnd.lens.intersection((ray.origin, ray.direction)).ok_or(RayTraceError::NoSurfaceIntersection)?;
                    ray = self.refract(p, normal, n1, n2, cmpnd.ar_coated)?;
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
pub(crate) fn detector_array_positioning<F: Float>(
    cmpnd: &CompoundPrism<F>,
    detarr: &LinearDetectorArray<F>,
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
    let l_vertex = lower_ray.direction.mul_add(d2, lower_ray.origin);
    let (pos, dir) = if d2 > F::zero() {
        (l_vertex, spec_dir)
    } else {
        let dists = imat * (-spec - upper_ray.origin + lower_ray.origin);
        let d2 = dists.y;
        if d2 < F::zero() {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let u_vertex = lower_ray.direction.mul_add(d2, lower_ray.origin);
        (u_vertex, -spec_dir)
    };
    Ok(DetectorArrayPositioning {
        position: pos,
        direction: dir,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spectrometer<F: Float> {
    pub gaussian_beam: GaussianBeam<F>,
    pub compound_prism: CompoundPrism<F>,
    pub detector_array: LinearDetectorArray<F>,
    pub detector_array_position: DetectorArrayPositioning<F>,
}

impl<F: Float> Spectrometer<F> {
    pub fn new(
        gaussian_beam: GaussianBeam<F>,
        compound_prism: CompoundPrism<F>,
        detector_array: LinearDetectorArray<F>,
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

    /// Propagate a ray of `wavelength` start `initial_y` through the spectrometer.
    /// Returning the intersection position on the detector array
    /// and the transmission probability.
    ///
    /// # Arguments
    ///  * `self` - spectrometer specification
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `initial_y` - the initial y value of the ray
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

    /// Trace the propagation of a ray of `wavelength` through the spectrometer.
    /// Returning an iterator of the ray's origin position and
    /// all of the intersection positions.
    ///
    /// # Arguments
    ///  * `self` - spectrometer specification
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `initial_y` - the initial y value of the ray
    #[cfg(not(target_arch = "nvptx64"))]
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

    pub(crate) fn size_and_deviation(&self) -> (F, F) {
        let deviation_vector = self.detector_array_position.position
            + self.detector_array_position.direction
                * self.detector_array.length
                * F::from_f64(0.5)
            - Pair {
                x: F::zero(),
                y: self.gaussian_beam.y_mean,
            };
        let size = deviation_vector.norm();
        let deviation = deviation_vector.y.abs() / deviation_vector.norm();
        (size, deviation)
    }

    pub(crate) fn probability_z_in_bounds(&self) -> F {
        let p_z = F::from_f64(crate::erf::erf(
            self.compound_prism.width.to_f64() * core::f64::consts::FRAC_1_SQRT_2
                / self.gaussian_beam.width.to_f64(),
        ));
        debug_assert!(F::zero() <= p_z && p_z <= F::one());
        p_z
    }
}
// cpu => 1.434454269122527

// 5.6923 ms & approx.ftz => 1.4349899
// 6.5894 ms & full.ftz => 1.4349899
// 34.771 ms & div.f32 => 1.4349926

// 180.14 ms & div.f64 => 1.4339791804027775

impl<F1: Float + LossyInto<F2>, F2: Float> LossyInto<GaussianBeam<F2>> for GaussianBeam<F1> {
    fn into(self) -> GaussianBeam<F2> {
        GaussianBeam {
            width: self.width.into(),
            y_mean: self.y_mean.into(),
            w_range: LossyInto::into(self.w_range),
        }
    }
}

impl<F1: Float + LossyInto<F2>, F2: Float> LossyInto<CompoundPrism<F2>> for CompoundPrism<F1> {
    fn into(self) -> CompoundPrism<F2> {
        CompoundPrism {
            prisms: LossyInto::into(self.prisms),
            lens: LossyInto::into(self.lens),
            height: self.height.into(),
            width: self.width.into(),
            ar_coated: self.ar_coated,
        }
    }
}

impl<F1: Float + LossyInto<F2>, F2: Float> LossyInto<LinearDetectorArray<F2>> for LinearDetectorArray<F1> {
    fn into(self) -> LinearDetectorArray<F2> {
        LinearDetectorArray {
            bin_count: self.bin_count.into(),
            bin_size: self.bin_size.into(),
            linear_slope: self.linear_slope.into(),
            linear_intercept: self.linear_intercept.into(),
            min_ci: self.min_ci.into(),
            angle: self.angle.into(),
            normal: LossyInto::into(self.normal),
            length: self.length.into(),
        }
    }
}

impl<F1: Float + LossyInto<F2>, F2: Float> LossyInto<DetectorArrayPositioning<F2>> for DetectorArrayPositioning<F1> {
    fn into(self) -> DetectorArrayPositioning<F2> {
        DetectorArrayPositioning {
            position: LossyInto::into(self.position),
            direction: LossyInto::into(self.direction),
        }
    }
}

impl<F1: Float + LossyInto<F2>, F2: Float> LossyInto<Spectrometer<F2>> for Spectrometer<F1> {
    fn into(self) -> Spectrometer<F2> {
        Spectrometer {
            gaussian_beam: LossyInto::into(self.gaussian_beam),
            compound_prism: LossyInto::into(self.compound_prism),
            detector_array: LossyInto::into(self.detector_array),
            detector_array_position: LossyInto::into(self.detector_array_position),
        }
    }
}
