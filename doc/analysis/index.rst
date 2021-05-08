.. _reports:

=====================================================
``analysis``: Model specification for BIDS datasets
=====================================================

.. contents:: **Contents**
    :local:
    :depth: 1

|

.. currentmodule:: pybids.modeling

.. _overview:

An overview of the analysis module
===========================================

.. topic:: **BIDS models**

  The `BIDS Models specification`_ is an extension to the BIDS standard for
  describing and organizing general linear models (GLMs) or similar models
  fitted to BIDS datasets.

  A GLM can be concisely described with a json file, regardless of the software
  used to fit the model (e.g., `nistats`_, `AfNI`_, `FSL`_, `SPM`_).

The PyBIDS analysis module provides high-level model specification functionality
for BIDS datasets. It assumes that model information is represented in line with
the (upcoming) BIDS-Model specification.

Note that, at present, pybids.modeling does not provide a one-stop model-fitting
solution. You will need to call some other package (e.g., `nistats`_, or a
non-Python package like `FSL`_ or `SPM`_ via `Nipype`_) to handle model estimation.
What pybids.modeling will do for you is automatically handle the loading and
transformation of all variables, and the construction of design matrices and
contrasts.

.. _BIDS Models specification: https://docs.google.com/document/d/1bq5eNDHTb6Nkx3WUiOBgKvLNnaa5OMcGtD0AZ9yms2M/edit#heading=h.mqkmyp254xh6
.. _nistats: https://nistats.github.io
.. _AfNI: https://afni.nimh.nih.gov
.. _FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
.. _SPM: http://www.fil.ion.ucl.ac.uk/spm/
.. _Nipype: http://nipype.readthedocs.io

.. _quickstart:

Quickstart
=============
A simple example of standard usage follows. We assume that we have a root folder
containing a BIDS-compliant project in ``/bidsproject``, as well as a BIDS-Models
JSON specification in ``model.json``.

    >>> from bids.modeling import BIDSStatsModelsGraph
    >>> # Initialize the BIDSStatsModelsGraph
    >>> analysis = BIDSStatsModelsGraph('/bidsproject', 'model1.json')
    >>> # Setup constructs all the design matrices
    >>> analysis.setup()
    >>> # Sample query: retrieve first-level design matrix for one run
    >>> dm = analysis[0].get_design_matrix(subject='01', run=1, task='taskA')
    >>> # Sample query: retrieve session-level contrast matrix
    >>> cm = analysis[1].get_contrasts(subject='01', session='retest')
