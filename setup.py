from setuptools import setup


def build_native(spec):
    build = spec.add_external_build(
        cmd=['cargo', 'build', '--release'],
        path='./'
    )

    spec.add_cffi_module(
        module_path='compound_prism_designer._native',
        dylib=lambda: build.find_dylib('compound_prism_designer', in_path='target/release'),
        header_filename=lambda: build.find_header('bindings.h', in_path='target'),
        rtld_flags=['NOW', 'NODELETE']
    )


setup(
    name="compound_prism_designer",
    version="0.0.3",
    packages=["compound_prism_designer"],
    zip_safe=False,
    platforms='any',
    setup_requires=["milksnake"],
    install_requires=["cffi", "numpy", "pygmo"],
    milksnake_tasks=[
        build_native
    ]
)
