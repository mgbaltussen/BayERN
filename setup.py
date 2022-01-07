from setuptools import setup, find_packages

with open('LICENSE') as f:
    license = f.read()

with open('README.md') as f:
    readme = f.read()

setup(
    name="BayERN",
    version="0.1.0",
    description="A Python package to perform Bayesian inference on enzyme kinetics data with PyMC3",
    long_description=readme,
    author="Mathieu G. Baltussen",
    author_email="m.g.baltussen@gmail.com",
    license=license,
    packages=find_packages(exclude=('test', 'docs')),
)
