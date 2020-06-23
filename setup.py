#!/usr/bin/env python3

import setuptools
import os

package_name = "mel"
packages = setuptools.find_packages(
    include=[package_name, "{}.*".format(package_name)]
)

# Version info -- read without importing
_locals = {}
with open(os.path.join(package_name, "_version.py")) as fp:
    exec(fp.read(), None, _locals)
version = _locals["__version__"]
binary_names = _locals["binary_names"]

# Frankenstein long_description: changelog note + README
long_description = """
To find out what's new in this version of mel, please see `the repo
<https://github.com/shawwn/mel>`_.

Welcome to mel!
=====================

`mel` is a Python library to simplify working with tensorflow and pytorch.
"""

setuptools.setup(
    name=package_name,
    version=version,
    description="Machine Learning toolkit",
    license="BSD",
    long_description=long_description,
    author="Shawn Presser",
    author_email="shawnpresser@gmail.com",
    url="https://github.com/shawwn/mel",
    install_requires=[
    ],
    packages=packages,
    entry_points={
        "console_scripts": [
            "{} = {}.program:cli".format(binary_name, package_name)
            for binary_name in binary_names
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
)

