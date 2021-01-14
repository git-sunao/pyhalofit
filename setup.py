import re
import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requires = ['scipy>=1.1.0', 'astropy>=3.2.3']

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
    url=find('url'),
    author=find('author'),
    author_email='sunao.sugiyama@ipmu.jp',
    keywords=['cosmology', 'large scale structure', 'halofit'],
    packages=['pyhalofit'],
    install_requires=requires,
    classifiers=['Programming Language :: Python :: 3.6'],
)
