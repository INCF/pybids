## PyBIDS: Modeling

The PyBIDS `modeling` module provides high-level model specification functionality for BIDS datasets. It assumes that model information is represented in line with the (upcoming) BIDS-Model specification.

Note that, at present, pybids.modeling does *not* provide a one-stop model-fitting solution. You will need to call some other package (e.g., nistats, or a non-Python package like FSL or SPM via Nipype) to handle model estimation. What pybids.modeling *will* do for you is automatically handle the loading and transformation of all variables, and the construction of design matrices and contrasts.
