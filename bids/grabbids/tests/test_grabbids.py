""" Tests of BIDS-specific functionality. Generic tests of core grabbit
functionality should go in the grabbit package. """

import pytest
from bids.grabbids import BIDSLayout
from os.path import join, dirname


# Fixture uses in the rest of the tests
@pytest.fixture
def testlayout1():
    data_dir = join(dirname(__file__), 'data', '7t_trt')
    return BIDSLayout(data_dir)


def test_layout_init(testlayout1):
    assert isinstance(testlayout1.files, dict)


def test_get_metadata(testlayout1):
    target = 'sub-03/ses-2/func/sub-03_ses-2_task-' \
             'rest_acq-fullbrain_run-2_bold.nii.gz'
    result = testlayout1.get_metadata(join(testlayout1.root, target))
    assert result['RepetitionTime'] == 3.0


def test_get_metadata2(testlayout1):
    target = 'sub-03/ses-1/fmap/sub-03_ses-1_run-1_phasediff.nii.gz'
    result = testlayout1.get_metadata(join(testlayout1.root, target))
    assert result['EchoTime1'] == 0.006


def test_get_subjects(testlayout1):
    result = testlayout1.get_subjects()
    predicted = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    assert predicted == result


def test_get_fieldmap(testlayout1):
    target = 'sub-03/ses-1/func/sub-03_ses-1_task-' \
             'rest_acq-fullbrain_run-1_bold.nii.gz'
    result = testlayout1.get_fieldmap(join(testlayout1.root, target))
    assert result["type"] == "phasediff"
    assert result["phasediff"].endswith('sub-03_ses-1_run-1_phasediff.nii.gz')


def test_get_fieldmap2(testlayout1):
    target = 'sub-03/ses-2/func/sub-03_ses-2_task-' \
             'rest_acq-fullbrain_run-2_bold.nii.gz'
    result = testlayout1.get_fieldmap(join(testlayout1.root, target))
    assert result["type"] == "phasediff"
    assert result["phasediff"].endswith('sub-03_ses-2_run-2_phasediff.nii.gz')


def test_bids_json(testlayout1):
    assert testlayout1.get(return_type='id', target='run') == ['1', '2']
    assert testlayout1.get(return_type='id', target='session') == ['1', '2']
