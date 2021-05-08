# PyBIDS
[![DOI](https://joss.theoj.org/papers/10.21105/joss.01294/status.svg)](https://doi.org/10.21105/joss.01294)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2537312.svg)](https://doi.org/10.5281/zenodo.2537312)

[![Build Status (Linux and OS X)](https://travis-ci.org/bids-standard/pybids.svg?branch=master)](https://travis-ci.org/bids-standard/pybids)<!---[![Build Status (Windows)](https://ci.appveyor.com/api/projects/status/5aa4c6e3m15ew4v7?svg=true)](https://ci.appveyor.com/project/chrisfilo/pybids-ilb80)-->
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/bids-standard/pybids/master)
[![PyPI version](https://badge.fury.io/py/pybids.svg)](https://badge.fury.io/py/pybids)

PyBIDS is a Python library to centralize interactions with datasets conforming
BIDS (Brain Imaging Data Structure) format.  For more information about BIDS
visit http://bids.neuroimaging.io.

PyBIDS currently supports Python 3 on POSIX operating systems (including Mac OS).  Windows is not officially supported, though most PyBIDS functionality will probably work fine.

## Installation
PyBIDS is most easily installed from pip. To install the latest official release:

> pip install pybids

If you want to live on the bleeding edge, you can install from master:

> pip install git+https://github.com/bids-standard/pybids.git

#### Dependencies
PyBIDS has a number of dependencies. The core querying functionality requires only the `BIDS-Validator` package. However, most other modules require the core Python neuroimaging stack: `numpy`, `scipy`, `pandas`, and `nibabel`. The `reports` module additionally requires `num2words`. By default, all dependencies will be installed with pybids (if they aren't already available).

## Usage
Get started by checking out [the documentation](https://bids-standard.github.io/pybids)!

Or you can start at [our tutorial](examples/pybids_tutorial.ipynb)! You can run it interactively without installing anything via [binder](https://mybinder.org/v2/gh/bids-standard/pybids/master). Click on the link and then navigate to `examples/pybids_tutorial.ipynb` to explore.

## How to cite
To credit PyBIDS in your work, please cite both the [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.01294) and the [Zenodo archive](https://zenodo.org/record/3363985). The former provides a high level description of the package, and the latter points to a permanent record of all PyBIDS versions (we encourage you to cite the specific version you used). Example citations (for PyBIDS 0.9.3):

Yarkoni et al., (2019). PyBIDS: Python tools for BIDS datasets. Journal of Open Source Software, 4(40), 1294, https://doi.org/10.21105/joss.01294

Yarkoni, Tal, Markiewicz, Christopher J., de la Vega, Alejandro, Gorgolewski, Krzysztof J., Halchenko, Yaroslav O., Salo, Taylor, … Blair, Ross. (2019, August 8). bids-standard/pybids: 0.9.3 (Version 0.9.3). Zenodo. http://doi.org/10.5281/zenodo.3363985

## Community guidelines
Bug reports, feedback, and other contributions to PyBIDS are welcome. To report a bug, request a feature, or raise a technical (non-usage-related) question for discussion, please [open an issue on GitHub](https://github.com/bids-standard/pybids/issues). Requests for support and questions about usage (where no bug is suspected) should be posted on the [NeuroStars](https://neurostars.org) forum. For questions about the [BIDS specification](https://github.com/bids-standard/bids-specification) or [BIDS-Validator](https://github.com/bids-standard/bids-validator), please see the corresponding repositories. For general BIDS-related discussion, see the [bids-discussion](https://groups.google.com/forum/#!forum/bids-discussion) mailing list.

Pull requests are always welcome. If you have any doubts about the suitability of a PR, or want to discuss implementation-related issues, feel free to open an issue for discussion.
