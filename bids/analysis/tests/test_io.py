from bids.grabbids import BIDSLayout
from bids.analysis.variables import (SparseEventVariable, SimpleVariable,
                                     load_variables)
from bids.analysis.variables.base import Run, Dataset
import pytest
from os.path import join, dirname, abspath
from bids import grabbids


@pytest.fixture
def layout():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    return BIDSLayout(path)


@pytest.fixture(scope="module")
def layout2():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', '7t_trt')
    layout = BIDSLayout(path)
    return layout


def test_load_events(layout1):
    dataset = load_variables(layout1, 'events', scan_length=480)
    runs = dataset.get_runs(subject='01')
    assert len(runs) == 3
    assert isinstance(runs[0], Run)
    variables = runs[0].variables
    assert len(variables) == 10
    targ_cols = {'parametric gain', 'PTval', 'trial_type', 'respnum'}
    assert not (targ_cols - set(variables.keys()))
    assert isinstance(variables['parametric gain'], SparseEventVariable)
    assert variables['parametric gain'].entities.shape == (86, 4)


def test_load_physio(layout2):
    pass


def test_load_participants(layout1):
    dataset = load_variables(layout1, 'participants')
    assert isinstance(dataset, Dataset)
    assert len(dataset.variables) == 2
    assert {'age', 'sex'} == set(dataset.variables.keys())
    age = dataset.variables['age']
    assert isinstance(age, SimpleVariable)
    assert age.entities.shape == (16, 1)
    assert age.values.shape == (16,)
