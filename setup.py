from setuptools import setup, find_packages

with open('LICENSE') as f:
    license = f.read()

with open('README.md') as f:
    readme = f.read()

setup(
    name="BayERN",
    version="0.3.0dev",
    description="A Python package to perform Bayesian inference on enzyme kinetics data with PyMC3",
    long_description=readme,
    packages=['bayern'],
    author="Mathieu G. Baltussen",
    author_email="m.g.baltussen@gmail.com",
    license=license,
)
