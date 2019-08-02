# Compound Prism Spectrometer Designer
Designer for an ultra-fast high-efficiency broad-spectrum low-resolution spectrometer with a [compound prism](https://en.wikipedia.org/wiki/Compound_prism) 
as the dispersive element. The optimization and design of a spectrometer has multiple conflicting goals; the size 
of the spectrometer and the light's deviation should be minimized, and the amount of 
[information](https://en.wikipedia.org/wiki/Quantities_of_information) should be maximised. 
Normal optimization methods that give one best result, can't be used with these goals, so multi-objective-optimization 
algorithms are used to generate a [Pareto Set](https://en.wikipedia.org/wiki/Pareto_efficiency#Pareto_frontier) 
of results. The results are then shown interactively to allow the user to choose which design they want.

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
  * Use the default stable toolchain
  * Need rust version >= 1.35.0
2. Create and activate a python virtual environment, using either
  - Python3 venv
    * Create and activate python3 virtual environment. See [tutorial](https://docs.python.org/3/tutorial/venv.html).
    * Install the prerequisite packages in the virtual environment with ``pip3 install -r requirements.txt``
  - Conda (Use if not on Linux)
    * Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
        for python virtual environment package environment.
    * Create and activate a conda virtual environment using [environment.yml](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) 
3. Build the library and add it to your using virtual environment with ``pip install -e .``
4. Configure design specifications in ``design_config.toml``
5. Run the designer ``optimizer.py``

## Design
```
h ≡ prism height
l_z ≡ prism width
l_a ≡ detector array length
w_beam ≡ 1 / e^2 beam width
T(λ, y) ≡ transmittance probability of a wavelength from an inital y, to the intersection with the detector array
S(λ, y) ≡ intersection position on the plane of the detector array, of a wavelength from an inital y
D ≡ { d ∈ detectors with [lb_d, ub_d) ⊆ [0, l_a] }
Λ ≡ [λ_min, λ_max]
Y ≡ (0, h)
Z ≡ (-l_z / 2, l_z / 2)
f(y, z) ≡ Exp[-2(y^2 + z^2)/w_beam^2] * 2 / (π * w_beam^2)
p(D=d|Λ=λ ∩ Y=y) ≡ T(λ, y) if lb_d <= ((S(λ, y) - det_arr_pos) • det_arr_dir) < ub_d else 0
p(Λ=λ) ≡ 1 / (λ_max - λ_min)
---
p(D=d|Λ=λ) = Integrate[p(D=d|Λ=λ ∩ Y=y) * Integrate[f(y - y_mean, z), z ∈ Z], y ∈ Y]
p(D=d) = Integrate[p(Λ=λ) p(D=d|Λ=λ), λ ∈ Λ]
dev_vector = det_arr_pos + det_arr_dir * l_a / 2 - (0, y_mean)
H(D) = -Sum[p(D=d) Log[p(D=d)], d ∈ D]
H(D|Λ) = -Sum[Integrate[p(Λ=λ) p(D=d|Λ=λ) Log[p(D=d|Λ=λ)], λ ∈ Λ], d ∈ D]
---
size = ||dev_vector||
dev = | dev_vector • (0, 1) | / ||dev_vector||
info = I(Λ; D) ≡ H(D) - H(D|Λ)
```