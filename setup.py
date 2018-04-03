#!/usr/bin/env python

from setuptools import setup

setup(
    name='derplog',
    version='0.1',
    description='DeepLog',
    author='Mike McClurg',
    author_email='mike.mcclurg@gmail.com',
    url='https://github.com/mcclurmc/derplog',
    packages=['derplog'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
