from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="prism",
    version="0.0.1",
    rust_extensions=[RustExtension("prism._prism", debug=False, binding=Binding.RustCPython)],
    packages=["prism"],
    install_requires=["numpy", "pygmo"],
    zip_safe=False,
)
