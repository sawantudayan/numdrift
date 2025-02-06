"""setup file to define the package for PyPI"""

from setuptools import setup, find_packages

setup(
    name='numdrift',
    version='0.1.0',
    description='A high-performance, memory-efficient library for numeric data manipulation.',
    author='Udayan Sawant',
    author_email='udayansawant7@gmail.com',
    url='https://github.com/sawantudayan/numdrift',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'numba',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
