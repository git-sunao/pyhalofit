import re
import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def get_long_description():
    with open(os.path.join('docs/source', 'README_pypi.rst')) as f:
        return f.read()

def get_requirements():
    fname = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(fname, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements

def find(what='version'):
    f = open(os.path.join(os.path.dirname(__file__), 'pyhalofit/__init__.py')).read()
    match = re.search(r"^__%s__ = ['\"]([^'\"]*)['\"]"%(what), f, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find %s string."%what)


setup(
    name='pyhalofit',
    version=find('version'),
    description='pyhalofit package',
    long_description='pyhalofit',#get_long_description(),
    url=find('url'),
    author=find('author'),
    author_email='sunao.sugiyama@ipmu.jp',
    keywords=['cosmology', 'large scale structure', 'halofit'],
    packages=['pyhalofit'],
    install_requires=get_requirements(),
    classifiers=['Programming Language :: Python :: 3.6'],
)
