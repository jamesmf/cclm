import os
from setuptools import setup

setup(
    name="cclm",
    version="0.0.1",
    author="jamesmf",
    author_email="",
    description=("composable character level models"),
    license="BSD",
    keywords="embeddings composable character-level",
    url="",
    packages=["cclm"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=[
        "tokenizers",
        "datasets",
        "tensorflow-gpu",
        "numpy",
        "tqdm",
        "pytest",
    ],
)
