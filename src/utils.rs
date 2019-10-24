use crate::ray::*;
use svg::node::element::path::Parameters;

impl Into<svg::node::element::path::Parameters> for Pair {
    fn into(self) -> Parameters {
        (self.x, self.y).into()
    }
}

pub fn create_svg(
    cmpnd: &CompoundPrism,
    detarr: &DetectorArray,
    detpos: &DetectorArrayPositioning,
    beam: &GaussianBeam,
) -> svg::Document {
    let mut document = svg::Document::new();
    // Draw Compound Prism
    let (polys, lens_poly, theta, lens_radius) = cmpnd.polygons();
    let poly_len = polys.len();
    // - Draw Non-Lens-Like prism trapezoids
    for (i, poly) in polys.into_iter().enumerate() {
        let [p0, p1, p2, p3] = poly;
        let data = svg::node::element::path::Data::new()
            .move_to(p0)
            .line_to(p1)
            .line_to(p2)
            .line_to(p3)
            .close();
        let path = svg::node::element::Path::new()
            .set("fill", if i % 2 == 0 { "white" } else { "gray" })
            .set("stroke", "black")
            .set("stroke-width", 1)
            .set("d", data);
        document = document.add(path);
    }
    // - Draw Lens-Like prism trapezoid
    let [p0, p1, p2, p3] = lens_poly;
    let data = svg::node::element::path::Data::new()
        .move_to(p0)
        .line_to(p1)
        .line_to(p2)
        .elliptical_arc_to((
            lens_radius,                                       // rx
            lens_radius,                                       // ry
            0.,                                                // x-axis rotation
            if theta >= std::f64::consts::PI { 1 } else { 0 }, // large-arc-flag
            0,                                                 // sweep-flag
            p3.x,
            p3.y,
        ))
        .close();
    let path = svg::node::element::Path::new()
        .set("fill", if poly_len % 2 == 0 { "white" } else { "gray" })
        .set("stroke", "black")
        // .set("stroke-width", 0.5)
        .set("d", data);
    document = document.add(path);
    // Draw Detector array
    let (s, e) = detarr.end_points(detpos);
    let data = svg::node::element::path::Data::new().move_to(s).line_to(e);
    let path = svg::node::element::Path::new()
        .set("fill", "none")
        .set("stroke", "black")
        // .set("stroke-width", 0.5)
        .set("d", data);
    document = document.add(path);
    // Draw beam path
    let (wmin, wmax) = beam.w_range;
    let wgrad = |p| wmin + p * (wmax - wmin);
    let ws = [wmin, wgrad(0.25), wgrad(0.5), wgrad(0.75), wmax];
    // http://colorbrewer2.org/#type=sequential&scheme=PuRd&n=5
    let wcolor = ["#f1eef6", "#d7b5d8", "#df65b0", "#dd1c77", "#980043"];
    let ys = [
        beam.y_mean - beam.width * 0.5,
        beam.y_mean,
        beam.y_mean + beam.width * 0.5,
    ];
    for (w, c) in ws.iter().zip(wcolor.iter().copied()) {
        for y in ys.iter() {
            if let Ok(path) = trace(*w, *y, &cmpnd, &detarr, *detpos).collect::<Result<Vec<_>, _>>()
            {
                let mut data = svg::node::element::path::Data::new();
                data = data.move_to(path[0]);
                for p in path[1..].iter().copied() {
                    data = data.line_to(p);
                }
                let path = svg::node::element::Path::new()
                    .set("fill", "none")
                    .set("stroke", c)
                    .set("stroke-width", 0.1)
                    .set("d", data);
                document = document.add(path);
            }
        }
    }
    // create bounding box
    let min_x = s.x.min(e.x).min(0.);
    let min_y = s.y.min(e.y).min(0.);
    let max_x = s.x.max(e.x).max(p2.x).max(p3.x);
    let max_y = s.y.max(e.y).max(p2.y).max(p3.y);
    document
        .set("background-fill", "white")
        .set(
            "transform",
            format!("matrix(1 0 0 -1 {} {})", -min_x, max_y),
        )
        .set("viewBox", (0, 0, max_x - min_x, max_y - min_y))
}

#[derive(Debug, Clone, Default)]
struct ZemaxCOORDBRKSurface {
    decenter_x: f64,
    decenter_y: f64,
    tilt_about_x: f64,
    tilt_about_y: f64,
    tilt_about_z: f64,
    order: f64
}

#[derive(Debug, Clone)]
enum ZemaxSurfaceType {
    STANDARD(),
    COORDBRK {
        decenter_x: f64,
        decenter_y: f64,
        tilt_about_x: f64,
        tilt_about_y: f64,
        tilt_about_z: f64,
        order: f64
    },
    TILTSURF { x_tangent: f64, y_tangent: f64 },
    BICONICX { x_radius: f64 },
}

impl Default for ZemaxSurfaceType {
    fn default() -> Self {
        ZemaxSurfaceType::STANDARD()
    }
}

impl ZemaxSurfaceType {
    fn name(&self) -> &'static str {
        match self {
            ZemaxSurfaceType::STANDARD() => "STANDARD",
            ZemaxSurfaceType::COORDBRK{..} => "COORDBRK",
            ZemaxSurfaceType::TILTSURF{..} => "TILTSURF",
            ZemaxSurfaceType::BICONICX{..}=> "BICONICX",
        }
    }

    fn params(&self) -> [f64; 6] {
        let mut arr = [0_f64; 6];
        match self {
            ZemaxSurfaceType::COORDBRK {
                decenter_x,
                decenter_y,
                tilt_about_x,
                tilt_about_y,
                tilt_about_z,
                order
            } => {
                arr[0] = *decenter_x;
                arr[1] = *decenter_y;
                arr[2] = *tilt_about_x;
                arr[3] = *tilt_about_y;
                arr[0] = *tilt_about_z;
                arr[5] = *order;
            },
            ZemaxSurfaceType::TILTSURF{x_tangent, y_tangent} => {
                arr[0] = *x_tangent;
                arr[1] = *y_tangent;
            },
            ZemaxSurfaceType::BICONICX{ x_radius }=> {
                arr[0] = *x_radius;
            },
            _ => {}
        }
        arr
    }
}

#[derive(Debug, Clone, Default)]
struct ZemaxSurface<'s> {
    surface_number: usize,
    surface_type: ZemaxSurfaceType,
    thickness: f64,
    semi_diameter: f64,
    radius: f64,
    glass: Option<&'s str>,
}

impl<'s> ZemaxSurface<'s> {
    fn zmx_block(&self, fmt: &mut dyn std::io::Write) -> std::io::Result<()> {
        writeln!(fmt, "SURF {surface_number}", surface_number=self.surface_number)?;
        writeln!(fmt, "  TYPE {surface_type}", surface_type=self.surface_type.name())?;
        writeln!(fmt, "  FIMP\n")?;
        writeln!(fmt, "  CURV {radius} 0 0.0 0.0 0", radius=self.radius)?;
        writeln!(fmt, "  HIDE 0 0 0 0 0 0 0 0 0 0")?;
        writeln!(fmt, "  MIRR 2 0")?;
        writeln!(fmt, "  SLAB 0")?;
        for (i, p) in self.surface_type.params().iter().enumerate() {
            writeln!(fmt, "  PARM {} {}", 1 + i, p)?;
        }
        writeln!(fmt, "  DISZ {thickness}", thickness=self.thickness)?;
        if let Some(name) = self.glass {
            writeln!(fmt, "  GLAS {glass} 0 0 1.5 40 0 0 0 0 0 0", glass=name)?;
        }
        writeln!(fmt, "  DIAM {} 1 0 0 1 \"\"", self.semi_diameter)?;
        writeln!(fmt, "  MEMA {} 0 0 0 1 \"\"", self.semi_diameter)?;
        writeln!(fmt, "  POPS 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0")?;
        if self.semi_diameter > 0. {
            writeln!(fmt, "  FLAP 0 {} 0", self.semi_diameter)?;
        }
        Ok(())
    }
}

pub fn create_zmx<'s, I: Iterator<Item = &'s str>>(
    cmpnd: &CompoundPrism,
    detarr: &DetectorArray,
    detpos: &DetectorArrayPositioning,
    beam: &GaussianBeam,
    glass_names: I,
    fmt: &mut dyn std::io::Write
) -> std::io::Result<()> {
    let ytans = cmpnd
        .prisms
        .iter()
        .map(|(_, s)| s.normal.y / s.normal.x);
    let thickness = cmpnd
        .prisms[1..]
        .iter()
        .map(|(_, s)| s.midpt.x)
        .chain(std::iter::once(cmpnd.lens.chord.midpt.x))
        .scan(cmpnd.prisms[0].1.midpt.x, |x0, x1| {
            Some(x1 - std::mem::replace(x0, x1))
        });
    writeln!(fmt, r###"VERS 181119 693 105780 L105780
MODE SEQ
NAME
PFIL 0 0 0
LANG 0
UNIT MM X W X CM MR CPMM
ENPD {beam_width}
ENVD 20 1 0
GFAC 0 0
GCAT SCHOTT
RAIM 0 0 1 1 0 0 0 0 0 1
PUSH 19.88873953731639 117.51261512638798 0 0 0 0
SDMA 0 0 0
OMMA 0 0
FTYP 0 0 1 24 0 0 0 1
ROPD 2
HYPR 1
PICB 1
XFLN 0
YFLN 0
FWGN 1
VDXN 0
VDYN 0
VCXN 0
VCYN 0
VANN 0"###, beam_width=beam.width)?;

    let (wmin, wmax) = beam.w_range;
    for i in 0..24 {
        let w = wmin + (wmax - wmin) * (i as f64) / 23.;
        writeln!(fmt, "WAVM {} {} 1", 1 + i, w)?;
    }

    writeln!(fmt, r###"PWAV 13
POLS 1 0 1 0 0 1 0
GLRS 2 0
GSTD 0 100.000 100.000 100.000 100.000 100.000 100.000 0 1 1 0 0 1 1 1 1 1 1
NSCD 100 500 0 0.001 10 9.9999999999999995e-07 0 0 0 0 0 0 1000000 0 2
COFN QF "COATING.DAT" "SCATTER_PROFILE.DAT" "ABG_DATA.DAT" "PROFILE.GRD"
COFN COATING.DAT SCATTER_PROFILE.DAT ABG_DATA.DAT PROFILE.GRD
SURF 0
  TYPE STANDARD
  FIMP

  CURV 0.0 0 0 0 0 ""
  HIDE 0 0 0 0 0 0 0 0 0 0
  MIRR 2 0
  SLAB 12
  DISZ INFINITY
  DIAM 0 0 0 0 1 ""
  MEMA 0 0 0 0 1 ""
  POPS 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0"###)?;

    let mut i = 1;
    let mut count = || {
        let o = i;
        i += 1;
        o
    };
    macro_rules! next_surface {
        ($($inner:tt)*) => {
            ZemaxSurface {
                surface_number: count(),
                $($inner)*,
                ..ZemaxSurface::default()
            }.zmx_block(fmt)?;
        }
    }

    next_surface! {
        surface_type: ZemaxSurfaceType::COORDBRK {
            decenter_x: 0_f64,
            decenter_y: cmpnd.height * 0.5 - beam.y_mean,
            tilt_about_x: 0_f64,
            tilt_about_y: 0_f64,
            tilt_about_z: 0_f64,
            order: 0_f64,
        }
    }
    for ((g, y), t) in glass_names.zip(ytans).zip(thickness) {
        next_surface! {
            thickness: t,
            semi_diameter: cmpnd.height * 0.5,
            glass: Some(g),
            surface_type: ZemaxSurfaceType::TILTSURF {x_tangent: 0_f64, y_tangent: -y}
        }
    }
    let lens_tilt = ((-cmpnd.lens.chord.normal.y).asin()).to_degrees();
    next_surface! {
        surface_type: ZemaxSurfaceType::COORDBRK {
            decenter_x: 0_f64,
            decenter_y: 0_f64,
            tilt_about_x: -lens_tilt,
            tilt_about_y: 0_f64,
            tilt_about_z: 0_f64,
            order: 0_f64,
        }
    }
    let (u, l) = cmpnd.lens.chord.end_points(cmpnd.height);
    let chord_length = (u - l).norm();
    next_surface! {
        radius: -cmpnd.lens.radius,
        semi_diameter: chord_length * 0.5,
        surface_type: ZemaxSurfaceType::BICONICX { x_radius: 0_f64}
    }
    next_surface! {
        surface_type: ZemaxSurfaceType::COORDBRK {
            decenter_x: 0_f64,
            decenter_y: 0_f64,
            tilt_about_x: lens_tilt,
            tilt_about_y: 0_f64,
            tilt_about_z: 0_f64,
            order: 0_f64,
        }
    }
    let detarr_offset = (detpos.position + detpos.direction * detarr.length * 0.5) - cmpnd.lens.chord.midpt;
    next_surface! {
        thickness: detarr_offset.x,
        surface_type: ZemaxSurfaceType::COORDBRK {
            decenter_x: 0_f64,
            decenter_y: detarr_offset.y,
            tilt_about_x: 0_f64,
            tilt_about_y: 0_f64,
            tilt_about_z: 0_f64,
            order: 0_f64,
        }
    }
    next_surface! {
        surface_type: ZemaxSurfaceType::COORDBRK {
            decenter_x: 0_f64,
            decenter_y: 0_f64,
            tilt_about_x: -detarr.angle.to_degrees(),
            tilt_about_y: 0_f64,
            tilt_about_z: 0_f64,
            order: 0_f64,
        }
    }
    next_surface! {
        semi_diameter: detarr.length * 0.5,
        surface_type: ZemaxSurfaceType::STANDARD()
    }
    writeln!(fmt, r###"BLNK
BLNK
TOL TOFF   0   0 0.0000000000000000E+00 0.0000000000000000E+00   0 0 0 0 0
MNUM 1 1
MOFF   0   1 "" 0 0 0 1 1 0 0.0 "" 0"###)
}
