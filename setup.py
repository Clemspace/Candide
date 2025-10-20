#!/usr/bin/env python3
"""Setup script for Ramanujan Transformer."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="ramanujan-transformer",
    version="0.1.0",
    description="Efficient sparse transformers using Ramanujan graph theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Clément Castellon",
    author_email="clement.castellon@gmail.com",
    url="https://github.com/Clemspace/ramanujan-transformer",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies - torch will be installed separately via pip with CUDA support
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "tokenizers>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "logging": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)