from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "requests>=2", "librosa", "nolds", "pylab", "entropy", "math", "matplotlib", "scipy", "numpy>=1.16", "vmo"]

setup(
    name="NonlinearDynamics-VMO",
    version="0.0.1",
    author="Pauline Maouad",
    author_email="pauline.mouawad@lau.edu.lb",
    description="A toolbox for symbolic time series analysis of audio using the Variable Markov Oracle and Nonlinear Dynamics",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/nonlineardynamics-variablemarkovoracle/NonlinearDynamics-VMO/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        #"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
