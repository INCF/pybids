from .due import due, Doi
from .layout import BIDSLayout, BIDSLayoutIndexer

# For backwards compatibility
from bids_validator import BIDSValidator

__all__ = [
    "modeling",
    "BIDSLayout",
    "BIDSLayoutIndexer",
    "BIDSValidator",
    "config",
    "layout",
    "reports",
    "utils",
    "variables"
]

due.cite(Doi("10.1038/sdata.2016.44"),
         description="Brain Imaging Data Structure",
         tags=["reference-implementation"],
         path='bids')

del due, Doi

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
