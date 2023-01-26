import setuptools
import os
import io
import re

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

setuptools.setup(
    name='biobss',
    version=get_version(),
    author='OBSS',
    license='MIT',
    description='A biological signal processing library ...',
    long_description=get_long_description(),
    packages=setuptools.find_packages(),
    url="https://github.com/obss/BIOBSS",
    install_requires=get_requirements(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
