"""Setup script for TURL model code."""

from setuptools import setup, find_packages

setup(
    name="turl",
    version="0.1.0",
    description="TURL: Table Understanding through Representation Learning",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.0.0",
        "transformers",
        "numpy",
        "tqdm",
    ],
)
