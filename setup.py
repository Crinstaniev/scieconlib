from setuptools import setup, find_packages
from scieconlib import __version__

VERSION = __version__
DESCRIPTION = 'Tools for SciEcon projects'
LONG_DESCRIPTION = 'Tools for SciEcon projects'

# Setting up
setup(
    name='scieconlib',
    version=VERSION,
    author='Crinstaniev',
    author_email='zhuangzesen@gmail.com',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['plotly', 'numpy', 'pandas', 'scipy', 'tqdm'],
    # license='GPL',
    keywords=['python', 'game theory', 'economy'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
