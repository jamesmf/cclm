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
        "tokenizers==0.9.2",
        "datasets==1.1.2",
        "tensorflow-gpu==2.3.1",
        "numpy==1.18.5",
        "tqdm==4.49.0",
        "pytest==6.1.1",
        "keras==2.4.3",
    ],
)
