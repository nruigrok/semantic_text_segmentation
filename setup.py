#!/usr/bin/env python

from distutils.core import setup

setup(
    name="semantic_text_segmentation",
    version="1.3",
    description="Split a text into semantically coherent subtexts",
    author="Nel Ruigrok",
    author_email="nelruigrok@nieuwsmonitor.org",
    packages=["semantic_text_segmentation"],
    include_package_data=False,
    zip_safe=False,
    keywords=["API", "text"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Text Processing",
    ],
    install_requires=["sentence_transformers", "numpy"],
    extras_require={"dev": ["twine"]},
)
