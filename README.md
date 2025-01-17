# healpix-geometry-analysis

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/healpix-geometry-analysis?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/healpix-geometry-analysis/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/healpix-geometry-analysis/smoke-test.yml)](https://github.com/lincc-frameworks/healpix-geometry-analysis/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/lincc-frameworks/healpix-geometry-analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/lincc-frameworks/healpix-geometry-analysis)
[![Read The Docs](https://img.shields.io/readthedocs/healpix-geometry-analysis)](https://healpix-geometry-analysis.readthedocs.io/)

This project was developed for the LINCC-Frameworks projects [HATS](https://github.com/astronomy-commons/hats) and [LSDB](https://github.com/astronomy-commons/lsdb).

This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).

For more information about the project template see the 
[documentation](https://lincc-ppt.readthedocs.io/en/latest/).

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create env -n <env_name> python=3.10
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> pip install -e .'[dev]'
>> pre-commit install
>> conda install pandoc
```

Notes:
1. The single quotes around `'[dev]'` may not be required for your operating system.
2. `pre-commit install` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
3. Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)
