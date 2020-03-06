use crate::qrng::DynamicQrng;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus as PRng;

/// Simulated Binary Crossover Operator
#[derive(Debug)]
struct SBX {
    distribution_index: f64,
}

impl SBX {
    fn crossover(&self, x1: f64, x2: f64, lb: f64, ub: f64, rng: &mut PRng) -> f64 {
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
struct PM {
    probability: f64,
    distribution_index: f64,
}

impl PM {
    fn mutate(&self, x: f64, lb: f64, ub: f64, rng: &mut PRng) -> f64 {
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
pub struct Soln<F> {
    pub params: Vec<f64>,
    pub fitness: F,
}

pub enum ParetoDominance {
    Dominates,
    Dominated,
}

pub trait MultiObjectiveFitness: Clone {
    type Epsilons;

    fn pareto_dominace(&self, other: &Self) -> Option<ParetoDominance>;
    fn epsilon_dominance(&self, other: &Self, epsilons: &Self::Epsilons) -> Option<ParetoDominance>;
}

pub trait MultiObjectiveMinimizationProblem {
    type Fitness: MultiObjectiveFitness;

    fn epsilons(&self) -> &<Self::Fitness as MultiObjectiveFitness>::Epsilons;
    fn parameter_bounds(&self) -> Vec<(f64, f64)>;
    fn evaluate(&self, params: &[f64]) -> Option<Self::Fitness>;
}

fn add_to_archive<P: MultiObjectiveMinimizationProblem>(
    problem: &P,
    archive: &mut Vec<Soln<P::Fitness>>,
    child: Soln<P::Fitness>,
) -> bool {
    let mut dominated = false;
    archive.retain(
        |a| match a.fitness.epsilon_dominance(&child.fitness, problem.epsilons()) {
            Some(ParetoDominance::Dominates) => {
                dominated = true;
                true
            }
            Some(ParetoDominance::Dominated) => false,
            None => true,
        },
    );
    if !dominated {
        archive.push(child);
    }
    !dominated
}

pub fn optimize<P: MultiObjectiveMinimizationProblem>(
    problem: &P,
    iteration_count: usize,
    population_size: usize,
    offspring_size: usize,
    seed: u64,
    crossover_distribution_index: f64,
    mutation_distribution_index: f64,
    mutation_probability: f64,
) -> Vec<Soln<P::Fitness>> {
    let mut rng = PRng::seed_from_u64(seed);
    let variator = SBX {
        distribution_index: crossover_distribution_index,
    };
    let mutator = PM {
        distribution_index: mutation_distribution_index,
        probability: mutation_probability,
    };
    let bounds = problem.parameter_bounds();
    let qrng_seed = (&mut rng)
        .sample_iter(rand_distr::Uniform::new(0., 1.))
        .take(bounds.len())
        .collect();
    let mut qrng = DynamicQrng::new(qrng_seed);
    let mut population = Vec::with_capacity(population_size);
    while population.len() < population_size {
        let params = bounds
            .iter()
            .zip(qrng.next())
            .map(|((l, u), r)| l + (u - l) * r)
            .collect::<Vec<_>>();
        if let Some(fitness) = problem.evaluate(&params) {
            population.push(Soln { params, fitness });
        }
    }
    /*let mut archive = Vec::with_capacity(population_size);
    for p in population.iter() {
        add_to_archive(problem, &mut archive, p);
    }*/
    let mut archive = population;

    for _ in 0..iteration_count {
        let mut offspring = Vec::with_capacity(offspring_size);
        while offspring.len() < offspring_size {
            // TODO: Selector should be tournament based off crowding distance
            let len = archive.len();
            let i = rng.gen_range(0, len);
            let j = (i + rng.gen_range(0, len - 1)) % len;
            let p1 = &archive[i];
            let p2 = &archive[j];

            let params: Vec<_> = p1
                .params
                .iter()
                .zip(p2.params.iter())
                .zip(bounds.iter())
                .map(|((&x1, &x2), &(lb, ub))| {
                    let xnew = variator.crossover(x1, x2, lb, ub, &mut rng);
                    mutator.mutate(xnew, lb, ub, &mut rng)
                })
                .collect();
            if let Some(fitness) = problem.evaluate(&params) {
                offspring.push(Soln { params, fitness });
            }
        }
        for o in offspring.into_iter() {
            add_to_archive(problem, &mut archive, o);
        }
    }
    archive
}
