#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="autoscorer",
    version="0.0.1",
    author="Daniel Yaeger",
    author_email="yaeger@ohsu.edu",
    description="Automated scoring of sleep EMG",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/dbyaeger/autoscorer",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
