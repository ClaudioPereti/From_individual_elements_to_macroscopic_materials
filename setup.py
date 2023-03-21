import pathlib
from setuptools import setup
from setuptools import find_packages

# the difectory contaning this file
HERE = pathlib.Path(__file__).parent

# the text of the README file
README = (HERE / "README.md").read_text()


setup(
    name='AISC',
    version='0.0.1',
    package_dir={'': 'project_aisc/src'},
    packages=find_packages(),
    license='MIT',
    author='claudio',
    description='Data science project on superconductivity',
    long_description = README,
    long_description_content_type = "text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
