#!/usr/bin/env python3
"""Setup script for Candide Transformer."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="candide-transformer",
    version="0.1.0",
    description="Transparent and efficient transformer training with Ramanujan graph sparsity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Clément Castellon",
    author_email="clement.castellon@sorbonne-universite.fr",
    url="https://github.com/Clemspace/Candide",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "datasets>=2.14.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "safetensors>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "logging": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "all": [
            # Combines all extras
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="transformer, language-model, machine-learning, deep-learning, nlp, sparsity, ramanujan-graphs",
    project_urls={
        "Bug Reports": "https://github.com/Clemspace/Candide/issues",
        "Source": "https://github.com/Clemspace/Candide",
        "Documentation": "https://github.com/Clemspace/Candide/blob/main/README.md",
    },
)