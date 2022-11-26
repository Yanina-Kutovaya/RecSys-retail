#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name="recsys-retail",
    version="1.0",
    description="RecSys retail recommender system",
    author="Yanina Kutovaya",
    author_email="kutovaiayp@yandex.ru",
    url="https://github.com/Yanina-Kutovaya/RecSys-retail.git",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
