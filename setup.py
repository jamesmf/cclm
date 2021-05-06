import os
from setuptools import setup, find_packages

setup(
    name="cclm",
    version="0.0.1",
    author="jamesmf",
    author_email="",
    description=("composable character level models"),
    license="MIT",
    keywords="embeddings composable character-level",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=[
        "tokenizers",
        "datasets",
        "tensorflow",
        "numpy",
        "tqdm",
        "pytest",
    ],
)
