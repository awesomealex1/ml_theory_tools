from setuptools import setup, find_packages

setup(
    name="ml_theory_tools",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # List your package dependencies here
        # "requests>=2.25.1",
    ],
)
