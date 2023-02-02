import io
import os
import re

import setuptools


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "readme.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt", encoding="utf8") as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "biobss", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


_DEV_REQUIREMENTS = [
    "black==21.7b0",
    "deepdiff==5.5.0",
    "flake8==3.9.2",
    'importlib-metadata>=1.1.0,<4.3;python_version<"3.8"',
    "isort==5.9.2",
    "pytest>=7.0.1",
    "pytest-cov>=3.0.0",
    "pytest-timeout>=2.1.0",
    "click==8.0.4",
]


extras = {
    "dev": _DEV_REQUIREMENTS,
}

setuptools.setup(
    name="biobss",
    version=get_version(),
    author="OBSS R&D",
    license="MIT",
    description="A biological signal processing and feature extraction library.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["tests"]),
    url="https://github.com/obss/BIOBSS",
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require=extras,
    setup_requires=["pytest-runner"],
    test_suite="tests",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
    ],
    keywords="signal processing, feature extraction, photoplethysmography, PPG, electrocardiography, ECG, acceleration, electrodermal activity, EDA, galvanic skin response, HRV, Heart Rate Variability",
)
