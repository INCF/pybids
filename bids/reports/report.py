"""Generate publication-quality data acquisition methods section from BIDS
dataset.
"""
import json
import os.path as op
from collections import Counter

from bids.reports import utils
from bids.reports import parsing


class BIDSReport(object):
    """Generate publication-quality data acquisition section from BIDS dataset.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        Layout object for a BIDS dataset.
    config : :obj:`str` or :obj:`dict`, optional
        Configuration info for methods generation. Can be a path to a file
        (str), a dictionary, or None. If None, loads and uses default
        configuration information.
        Keys in the dictionary include:
            'dir':      a dictionary for converting encoding direction strings
                        (e.g., j-) to descriptions (e.g., anterior to
                        posterior)
            'seq':      a dictionary of sequence abbreviations (e.g., EP) and
                        corresponding names (e.g., echo planar)
            'seqvar':   a dictionary of sequence variant abbreviations
                        (e.g., SP) and corresponding names (e.g., spoiled)
    """

    def __init__(self, layout, config=None):
        self.layout = layout
        if config is None:
            config = op.join(
                op.dirname(op.abspath(__file__)), "config", "english", "converters.json"
            )

        if isinstance(config, str):
            with open(config) as fobj:
                config = json.load(fobj)

        if not isinstance(config, dict):
            raise ValueError(
                "Input config must be None, dict, or path to "
                "json file containing dict."
            )

        self.config = config

    def generate_from_files(self, files):
        r"""Generate a methods section from a list of files.

        Parameters
        ----------
        files : list of BIDSImageFile objects
            List of files from which to generate methods description.

        Returns
        -------
        counter : :obj:`collections.Counter`
            A dictionary of unique descriptions across subjects in the file list,
            along with the number of times each pattern occurred. In cases
            where all subjects underwent the same protocol, the most common
            pattern is most likely the most complete. In cases where the
            file list contains multiple protocols, each pattern will need to be
            inspected manually.

        Examples
        --------
        >>> from os.path import join
        >>> from bids.layout import BIDSLayout
        >>> from bids.reports import BIDSReport
        >>> from bids.tests import get_test_data_path
        >>> layout = BIDSLayout(join(get_test_data_path(), 'synthetic'))
        >>> report = BIDSReport(layout)
        >>> files = layout.get(session='01', extension=['.nii.gz', '.nii'])
        >>> counter = report.generate_from_files(files)
        Number of patterns detected: 1
        Remember to double-check everything and to replace <deg> with a degree symbol.

        >>> counter.most_common()[0][0]  # doctest: +ELLIPSIS
        'In session 01, MR data were...'
        """
        descriptions = []

        subjects = sorted(list(set([f.get_entities().get("subject") for f in files])))
        sessions = sorted(list(set([f.get_entities().get("session") for f in files])))
        for sub in subjects:
            subject_files = [f for f in files if f.get_entities().get("subject") == sub]
            description_list = []
            for ses in sessions:
                data_files = [
                    f for f in subject_files if f.get_entities().get("session") == ses
                ]

                if data_files:
                    ses_description = parsing.parse_files(
                        self.layout, data_files, sub, self.config, session=ses
                    )
                    ses_description[0] = ses_description[0].lower()
                    ses_description = "In session {0}, ".format(ses) + ses_description
                    description_list += ses_description
                    metadata = self.layout.get_metadata(data_files[0].path)
                else:
                    raise Exception("No imaging files for subject {0}".format(sub))

            # Assume all data were converted the same way and use the last nifti
            # file's json for conversion information.
            if "metadata" not in vars():
                raise Exception(
                    "No valid jsons found. Cannot generate final paragraph."
                )

            description = "\n\t".join(description_list)
            description += "\n\n{0}".format(parsing.final_paragraph(metadata))
            descriptions.append(description)
        counter = Counter(descriptions)
        print("Number of patterns detected: {0}".format(len(counter.keys())))
        print(utils.reminder())
        return counter

    def generate(self, **kwargs):
        r"""Generate the methods section.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to BIDSLayout to select subsets of the
            dataset.

        Returns
        -------
        counter : :obj:`collections.Counter`
            A dictionary of unique descriptions across subjects in the dataset,
            along with the number of times each pattern occurred. In cases
            where all subjects underwent the same protocol, the most common
            pattern is most likely the most complete. In cases where the
            dataset contains multiple protocols, each pattern will need to be
            inspected manually.

        Examples
        --------
        >>> from os.path import join
        >>> from bids.layout import BIDSLayout
        >>> from bids.reports import BIDSReport
        >>> from bids.tests import get_test_data_path
        >>> layout = BIDSLayout(join(get_test_data_path(), 'synthetic'))
        >>> report = BIDSReport(layout)
        >>> counter = report.generate(session='01')
        Number of patterns detected: 1
        Remember to double-check everything and to replace <deg> with a degree symbol.

        >>> counter.most_common()[0][0]  # doctest: +ELLIPSIS
        'In session 01, MR data were...'

        """
        descriptions = []

        subjects = self.layout.get_subjects(**kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k != "subject"}
        for sub in subjects:
            descriptions.append(self._report_subject(subject=sub, **kwargs))
        counter = Counter(descriptions)
        print("Number of patterns detected: {0}".format(len(counter.keys())))
        print(utils.reminder())
        return counter

    def _report_subject(self, subject, **kwargs):
        """Write a report for a single subject.

        Parameters
        ----------
        subject : :obj:`str`
            Subject ID.

        Attributes
        ----------
        layout : :obj:`bids.layout.BIDSLayout`
            Layout object for a BIDS dataset.
        config : :obj:`dict`
            Configuration info for methods generation.

        Returns
        -------
        description : :obj:`str`
            A publication-ready report of the dataset's data acquisition
            information. Each scan type is given its own paragraph.
        """
        description_list = []
        # Remove sess from kwargs if provided, else set sess as all available
        sessions = kwargs.pop(
            "session", self.layout.get_sessions(subject=subject, **kwargs)
        )
        if not sessions:
            sessions = [None]
        elif not isinstance(sessions, list):
            sessions = [sessions]

        for ses in sessions:
            data_files = self.layout.get(
                subject=subject, extension=[".nii", ".nii.gz"], **kwargs
            )

            if data_files:
                ses_description = parsing.parse_files(
                    self.layout, data_files, subject, self.config, session=ses
                )
                ses_description[0] = "In session {0}, ".format(ses) + ses_description[0]
                description_list += ses_description
                metadata = self.layout.get_metadata(data_files[0].path)
            else:
                raise Exception("No imaging files for subject {0}".format(subject))

        # Assume all data were converted the same way and use the first nifti
        # file's json for conversion information.
        description = "\n\t".join(description_list)
        description += "\n\n{0}".format(parsing.final_paragraph(metadata))
        return description
