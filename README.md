# Compound Prism Spectrometer Designer
Designer for an optimized spectrometer with a [compound prism](https://en.wikipedia.org/wiki/Compound_prism) 
as the dispersive element. The optimization and design of a spectrometer has multiple conflicting goals; the size 
of the spectrometer and the light's deviation should be minimized, and the amount of 
[information](https://en.wikipedia.org/wiki/Quantities_of_information)  should be maximised. 
Normal optimization methods that give one best result, can't be used with these goals, so multi-objective-optimization 
algorithms are used to generate a [Pareto Set](https://en.wikipedia.org/wiki/Pareto_efficiency#Pareto_frontier) 
of results. The results are then shown interactively to allow the user to choose which design they want.

## Design Assumptions
* Incoming light is [collimanted](https://en.wikipedia.org/wiki/Collimated_beam)
* Incoming light has a [gaussian intensity profile](https://en.wikipedia.org/wiki/Gaussian_beam)
* Incoming light has a uniform distribution of wavelengths
* Incoming light is unpolarized
* No intra-media transmission loss 
* Only inter-media transmission loss at media interfaces using
 [fresnel equations](https://en.wikipedia.org/wiki/Fresnel_equations)
* Ignores the effect of the light's incident angle with the detector on detection probability

## How to run
1. Install rust using [rustup](https://rustup.rs/), the rust installation manager
  * Use the default stable toolchain
  * Need rust version >= 1.35.0
2. Create and activate python3 virtual environment. See [tutorial](https://docs.python.org/3/tutorial/venv.html).
3. Install the prerequisite packages in the virtual environment with ``pip3 install -r requirements.txt``
3. Build the library and add it to your using virtual environment with ``pyo3-pack develop --release``
4. Configure design specifications in ``design_config.toml``
5. Run the designer ``optimizer.py``
