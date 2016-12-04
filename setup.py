# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='robustpg',
    version='0.0.1',
    description='Extensions to policy gradient methods',
    long_description=readme,
    author='Rishi Shah',
    author_email='rishihahs@gmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('plots')),
    entry_points={
        'console_scripts': [
            'robustpg = robustpg.__main__:main'
        ]
    },
)
