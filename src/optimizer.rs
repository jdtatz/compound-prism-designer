use crate::glasscat::Glass;
use crate::ray::*;
use binary_heap_plus::BinaryHeap;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256Plus as PRng;

#[derive(Debug, Clone)]
pub struct Params<'s> {
    prism_count: usize,
    prism_height: f64,
    glass_findices: &'s [f64],
    angles: &'s [f64],
    lengths: &'s [f64],
    curvature: f64,
    y_mean: f64,
    detector_array_angle: f64,
}

impl<'s> Params<'s> {
    pub fn from_slice(s: &'s [f64], max_prism_count: usize) -> Self {
        assert_eq!(s.len(), 3 * max_prism_count + 6);
        let (&prism_count, s) = s.split_first().unwrap();
        let (&prism_height, s) = s.split_first().unwrap();
        let (glass_findices, s) = s.split_at(max_prism_count);
        let (angles, s) = s.split_at(max_prism_count + 1);
        let (lengths, s) = s.split_at(max_prism_count);
        let (&curvature, s) = s.split_first().unwrap();
        let (&normalized_y_mean, s) = s.split_first().unwrap();
        let (&detector_array_angle, _) = s.split_first().unwrap();
        let prism_count = (prism_count as usize)
            .max(1)
            .min(max_prism_count);
        Self {
            prism_count,
            prism_height,
            glass_findices: &glass_findices[..prism_count],
            angles: &angles[..prism_count + 1],
            lengths: &lengths[..prism_count],
            curvature,
            y_mean: prism_height * normalized_y_mean,
            detector_array_angle,
        }
    }

    fn glass_indices(&self) -> impl 's + ExactSizeIterator<Item=usize> {
        self.glass_findices
            .iter()
            .map(|f| f.floor() as usize)
    }
}

/// Specification structure for the configuration of the Spectrometer Designer
#[derive(Constructor, Debug, Clone)]
pub struct Config {
    pub max_prism_count: usize,
    pub wavelength_range: (f64, f64),
    pub beam_width: f64,
    pub max_prism_height: f64,
    pub prism_width: f64,
    pub detector_array_length: f64,
    pub detector_array_min_ci: f64,
    pub detector_array_bin_bounds: Box<[[f64; 2]]>,
    pub glass_catalog: Box<[(String, Glass)]>,
}

impl Config {
    fn params_count(&self) -> usize {
        3 * self.max_prism_count + 6
    }

    fn param_bounds(&self) -> Box<[(f64, f64)]> {
        let prism_count_bounds = (1., (1 + self.max_prism_count) as f64);
        let prism_height_bounds = (0.001, self.max_prism_height);
        let glass_bounds = (0., (self.glass_catalog.len() - 1) as f64);
        let angle_bounds = (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
        let length_bounds = (0., self.max_prism_height);
        let curvature_bounds = (0.001, 1.0);
        let normalized_y_mean_bounds = (0., 1.0);
        let det_arr_angle_bounds = (-std::f64::consts::PI, std::f64::consts::PI);

        let mut bounds = Vec::with_capacity(self.params_count());
        bounds.push(prism_count_bounds);
        bounds.push(prism_height_bounds);
        for _ in 0..self.max_prism_count {
            bounds.push(glass_bounds)
        }
        for _ in 0..self.max_prism_count + 1 {
            bounds.push(angle_bounds)
        }
        for _ in 0..self.max_prism_count {
            bounds.push(length_bounds)
        }
        bounds.push(curvature_bounds);
        bounds.push(normalized_y_mean_bounds);
        bounds.push(det_arr_angle_bounds);
        bounds.into_boxed_slice()
    }

    pub fn array_to_params<'p, 's: 'p>(
        &'s self,
        params: &'p [f64],
    ) -> (
        CompoundPrism,
        impl 'p + Iterator<Item = &'s str>,
        DetectorArray,
        GaussianBeam,
    ) {
        assert_eq!(self.params_count(), params.len());
        let params = Params::from_slice(params, self.max_prism_count);
        let cmpnd = CompoundPrism::new(
            params.glass_indices().map(|i| &self.glass_catalog[i].1),
            &params.angles,
            &params.lengths,
            params.curvature,
            params.prism_height,
            self.prism_width,
        );
        let cat = &self.glass_catalog;
        let glass_names = params.glass_indices().map(move |i| cat[i].0.as_str());
        let detarr = DetectorArray::new(
            self.detector_array_bin_bounds.as_ref().into(),
            self.detector_array_min_ci,
            params.detector_array_angle,
            self.detector_array_length,
        );
        let beam = GaussianBeam {
            width: self.beam_width,
            y_mean: params.y_mean,
            w_range: self.wavelength_range,
        };
        (cmpnd, glass_names, detarr, beam)
    }

    fn evaluate(&self, params: &[f64]) -> Option<DesignFitness> {
        let (c, _, d, g) = self.array_to_params(params);
        fitness(&c, &d, &g)
            .ok()
            .filter(|f| f.size <= 30. * self.max_prism_height && f.info >= 0.1)
    }
}

fn approx_quality(a: &DesignFitness, p: &DesignFitness) -> f64 {
    (a.size - p.size)
        .max(p.info - a.info)
        .max(a.deviation - p.deviation)
}

fn min2_approx_quality<'a>(
    a: &DesignFitness,
    ps: impl Iterator<Item = (usize, &'a DesignFitness)>,
) -> [(usize, f64); 2] {
    // Steps 13 - 16
    let sh = ps.size_hint();
    let count = sh.1.unwrap_or(sh.0);
    let mut approx_heap =
        BinaryHeap::with_capacity_by(count, |(_, a): &(usize, f64), (_, p): &(usize, f64)| {
            p.partial_cmp(a).unwrap()
        });
    for (i, p) in ps {
        approx_heap.push((i, approx_quality(a, p)));
    }
    let min1 = approx_heap.pop().unwrap();
    let min2 = approx_heap.pop().unwrap();
    [min1, min2]
}

/// Simulated Binary Crossover Operator
#[derive(Debug)]
pub struct SBX {
    pub distribution_index: f64,
}

impl SBX {
    pub fn crossover(&self, x1: f64, x2: f64, lb: f64, ub: f64, rng: &mut PRng) -> f64 {
        let u: f64 = rng.gen_range(0., 1.);
        let beta = if u <= 0.5 {
            (2. * u).powf(1. / (self.distribution_index + 1.))
        } else {
            (1. / (2. * (1. - u))).powf(1. / (self.distribution_index + 1.))
        };
        let x1new = 0.5 * ((1. + beta) * x1 + (1. - beta) * x2);
        // let x2new = 0.5 * ((1. - beta) * x1 + (1. + beta) * x2);
        x1new.min(ub).max(lb)
    }
}

/// Polynomial Mutation
#[derive(Debug)]
pub struct PM {
    pub probability: f64,
    pub distribution_index: f64,
}

impl PM {
    pub fn mutate(&self, x: f64, lb: f64, ub: f64, rng: &mut PRng) -> f64 {
        let u: f64 = rng.gen_range(0., 1.);
        if u < self.probability {
            let u: f64 = rng.gen_range(0., 1.);
            let dx = ub - lb;
            let delta = if u <= 0.5 {
                (2. * u + (1. - 2. * u) * (1.0 - (x - lb) / dx).powf(self.distribution_index + 1.))
                    .powf(1. / (self.distribution_index + 1.))
                    - 1.
            } else {
                1. - (2. * (1. - u)
                    + 2. * (u - 0.5) * (1. - (ub - x) / dx).powf(self.distribution_index + 1.))
                .powf(1. / (self.distribution_index + 1.))
            };
            (x + delta * dx).min(ub).max(lb)
        } else {
            x
        }
    }
}

#[derive(Clone, Debug)]
pub struct Soln {
    pub params: Box<[f64]>,
    pub fitness: DesignFitness,
}

/// Approximation-Guided Evolutionary Multi-Objective Optimizer
/// https://www.ijcai.org/Proceedings/11/Papers/204.pdf
#[derive(Debug)]
pub struct AGE<'c> {
    population_size: usize,
    offspring_size: usize,
    problem: &'c Config,
    param_bounds: Box<[(f64, f64)]>,
    pub archive: Vec<Soln>,
    population: Vec<Soln>,
    rng: PRng,
    epsilons: [f64; 3],
    sbx: SBX,
    pm: PM,
}

impl<'c> AGE<'c> {
    pub fn new(
        problem: &'c Config,
        population_size: usize,
        offspring_size: usize,
        seed: u64,
        epsilons: [f64; 3],
        sbx: SBX,
        pm: PM,
    ) -> Self {
        let mut rng = PRng::seed_from_u64(seed);
        let bounds = problem.param_bounds();
        let mut population = Vec::with_capacity(population_size);
        while population.len() < population_size {
            let params = bounds
                .iter()
                .map(|(l, u)| rng.gen_range(l, u))
                .collect::<Box<_>>();
            if let Some(fitness) = problem.evaluate(&params) {
                population.push(Soln { params, fitness });
            }
        }
        let archive = population.clone();
        Self {
            problem,
            population_size,
            offspring_size,
            param_bounds: bounds,
            archive,
            population,
            rng,
            epsilons,
            sbx,
            pm,
        }
    }

    fn epsilon_dominance(eps: &[f64; 3], lhs: &DesignFitness, rhs: &DesignFitness) -> Option<bool> {
        let box_id = |f: &DesignFitness| {
            [
                (f.size / eps[0]).floor(),
                (f.info / eps[1]).ceil(),
                (f.deviation / eps[2]).floor(),
            ]
        };
        let square = |x: f64| x * x;
        let box_dist = |f: &DesignFitness, bid: &[f64; 3]| {
            square(f.size - bid[0] * eps[0])
                + square(bid[1] * eps[1] - f.info)
                + square(f.deviation - bid[2] * eps[2])
        };
        let l_box = box_id(lhs);
        let r_box = box_id(rhs);
        if l_box == r_box {
            let l_dist = box_dist(&lhs, &l_box);
            let r_dist = box_dist(&rhs, &r_box);
            if l_dist <= r_dist {
                Some(true)
            } else {
                Some(false)
            }
        } else if l_box[0] <= r_box[0] && l_box[1] >= r_box[1] && l_box[2] <= r_box[2] {
            Some(true)
        } else if l_box[0] >= r_box[0] && l_box[1] <= r_box[1] && l_box[2] >= r_box[2] {
            Some(false)
        } else {
            None
        }
    }

    fn add_to_archive(&mut self, child: &Soln) {
        let mut dominated = false;
        let eps = &self.epsilons;
        self.archive.retain(
            |a| match AGE::epsilon_dominance(eps, &a.fitness, &child.fitness) {
                Some(true) => {
                    dominated = true;
                    true
                }
                Some(false) => false,
                None => true,
            },
        );
        if !dominated {
            self.archive.push(child.clone());
        }
    }

    pub fn iterate(&mut self) {
        // Steps 4 to 8
        let mut offspring = Vec::with_capacity(self.offspring_size);
        while offspring.len() < self.offspring_size {
            // TODO: Selector should be tournament based off crowding distance
            let mut parents = self.population.choose_multiple(&mut self.rng, 2);
            let p1 = parents.next().unwrap();
            let p2 = parents.next().unwrap();

            let bounds = &self.param_bounds;
            let rng = &mut self.rng;
            let sbx = &self.sbx;
            let pm = &self.pm;
            let params: Box<_> = p1
                .params
                .iter()
                .zip(p2.params.iter())
                .zip(bounds.iter())
                .map(|((&x1, &x2), &(lb, ub))| {
                    let xnew = sbx.crossover(x1, x2, lb, ub, rng);
                    pm.mutate(xnew, lb, ub, rng)
                })
                .collect();
            if let Some(fitness) = self.problem.evaluate(&params) {
                offspring.push(Soln { params, fitness });
            }
        }
        // Steps 9 to 11
        for o in offspring.iter() {
            self.add_to_archive(o);
        }
        self.population.append(&mut offspring);
        // Steps 12 - 16
        let mut p1a1p2a2: Box<_> = self
            .archive
            .iter()
            .map(|a| {
                (min2_approx_quality(
                    &a.fitness,
                    self.population.iter().map(|s| &s.fitness).enumerate(),
                ))
            })
            .collect();
        // Steps 12 - 23
        let mut betas = BinaryHeap::with_capacity_by(
            self.archive.len(),
            |(_, kl): &(usize, f64), (_, kr): &(usize, f64)| kl.partial_cmp(&kr).unwrap(),
        );
        for (i, _) in self.population.iter().enumerate() {
            if let Some(a2) = p1a1p2a2
                .iter()
                .filter_map(|[(p1, _), (_, a2)]| if p1 == &i { Some(*a2) } else { None })
                .max_by(|a2l, a2r| a2l.partial_cmp(a2r).unwrap())
            {
                betas.push((i, a2))
            }
        }
        // Steps 12 - 23
        let mut removed_indices = vec![false; self.population.len()];
        let mut plen = self.population.len();
        while plen > self.population_size {
            let (p_star, _) = betas.pop().unwrap();
            if !removed_indices[p_star] {
                removed_indices[p_star] = true;
                plen -= 1;
                for (i, a) in self.archive.iter().enumerate() {
                    let [(p1, _), _] = p1a1p2a2[i];
                    if p1 == p_star {
                        let new = min2_approx_quality(
                            &a.fitness,
                            self.population
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| !removed_indices[*i])
                                .map(|(i, p)| (i, &p.fitness)),
                        );
                        betas.push(((new[0]).0, (new[1]).1));
                        p1a1p2a2[i] = new;
                    }
                }
            }
        }
        for (i, b) in removed_indices.into_iter().enumerate().rev() {
            if b {
                self.population.swap_remove(i);
            }
        }
    }
}
