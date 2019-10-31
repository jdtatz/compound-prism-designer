# Compound Prism Spectrometer Designer
Designer for an ultra-fast high-efficiency broad-spectrum low-resolution spectrometer with a [compound 
prism](https://en.wikipedia.org/wiki/Compound_prism) as the dispersive element. The optimization and design 
of a spectrometer has multiple conflicting goals; the size of the spectrometer and the light's deviation should be 
minimized, and the amount of [information](https://en.wikipedia.org/wiki/Quantities_of_information) should be maximised. 
Normal optimization methods that give one best result, can't be used with these goals, so multi-objective-optimization 
algorithms are used to generate a [Pareto Set](https://en.wikipedia.org/wiki/Pareto_efficiency#Pareto_frontier) 
of results. The results are then shown interactively to allow the user to choose which design they want.
The optimization algorithm used is [Approximation-Guided Evolutionary Multi-Objective 
Optimization II](https://cs.adelaide.edu.au/users/markus/pub/2013gecco-age2.pdf).

## Design Assumptions
* Incoming light is [collimanted](https://en.wikipedia.org/wiki/Collimated_beam)
* Incoming light has a [gaussian intensity profile](https://en.wikipedia.org/wiki/Gaussian_beam)
* Incoming light has a uniform distribution of wavelengths
* Incoming light is unpolarized
* The idea is that the incoming light will typically be from an [achromatic fiber-coupled collimator](https://www.thorlabs.com/navigation.cfm?guide_id=27)
* No intra-media transmission loss 
* Only inter-media transmission loss at media interfaces using
 [fresnel equations](https://en.wikipedia.org/wiki/Fresnel_equations)
* Ignores the effect of the light's incident angle with the detector on detection probability, other than the 
    idea that the detectors have a maximum allowed incident angle for efficiency reasons
* The detector array is linear
    * An example of one would be a linear ccd or a [pmt array](https://www.hamamatsu.com/resources/pdf/etd/LINEAR_PMT_TPMH1325E.pdf)
* The last prism is an aspheric chromatic [prism lens](https://en.wikipedia.org/wiki/Prism_correction), to allow 
 for wavelength-dependent focusing onto the detector array with minimal losses, 
 in comparision to a separate lens element.

## How to run
1. Install rust using [rustup](https://rustup.rs/), the rust installation manager
  * Use the non-default nightly toolchain
    - ``rustup toolchain install nightly``
    - And either
      * ``rustup override set nightly`` in the project directory
      * or ``rustup default nightly``
  * Need rust version >= 1.36.0
2. Create and activate a python virtual environment, using either
  - Python3 venv
    * Create and activate python3 virtual environment. See [tutorial](https://docs.python.org/3/tutorial/venv.html).
    * Install the prerequisite packages in the virtual environment with ``pip3 install -r requirements.txt``
  - Conda (Use if not on Linux)
    * Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
        for python virtual environment package environment.
    * Create and activate a conda virtual environment using [environment.yml](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) 
3. Build the library and add it to your using virtual environment with ``maturin develop --release --rustc-extra-args="-C target-feature=+fma -C target-cpu=native" --cargo-extra-args="--features pyext"``
4. Configure design specifications in ``design_config.toml``
5. Run the designer app with ``python3 -m compound_prism_designer``
