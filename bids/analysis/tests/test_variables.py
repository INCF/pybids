from bids.analysis.variables import (SparseEventColumn, load_variables,
                                     BIDSEventFile)
import pytest
from os.path import join, dirname, abspath
from bids import grabbids
import tempfile
import shutil


@pytest.fixture
def manager():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    return load_variables(path)


def test_load_all_bids_variables():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', '7t_trt')
    manager = load_variables(path, acq='fullbrain')


def test_clone_collection(manager):
    collection = manager['time']
    clone = collection.clone()
    assert clone != collection
    props_same = ['entities', 'default_duration', 'sampling_rate']
    for ps in props_same:
        assert getattr(collection, ps) is getattr(clone, ps)

    assert collection.columns.keys() == clone.columns.keys()
    assert collection.columns is not clone.columns
    assert collection.dense_index.equals(clone.dense_index)
    assert collection.dense_index is not clone.dense_index


def test_collection(manager):
    ''' Integration test for BIDSVariableCollection initialization. '''

    # Test that event files are loaded properly
    collection = manager['time']
    assert len(collection.event_files) == 48
    ef = collection.event_files[0]
    assert isinstance(ef, BIDSEventFile)
    assert ef.entities['task'] == 'mixedgamblestask'
    assert ef.entities['subject'] == '01'

    # Test extracted columns
    col_keys = collection.columns.keys()
    assert set(col_keys) == {'RT', 'gain', 'respnum', 'PTval', 'loss',
                             'respcat', 'parametric gain', 'trial_type'}
    col = collection.columns['RT']
    assert isinstance(col, SparseEventColumn)
    assert col.collection == collection
    assert col.name == 'RT'
    assert col.onset.max() == 476
    assert (col.duration == 3).all()
    assert len(col.duration) == 4096
    ents = col.entities
    assert (ents['task'] == 'mixedgamblestask').all()
    assert set(ents.columns) == {'task', 'subject', 'run', 'event_file_id'}


def test_read_from_files():

    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')

    path2 = join(dirname(abspath(grabbids.__file__)), 'tests', 'data', 'ds005')
    subs = ['02', '06', '08']
    template = 'sub-%s/func/sub-%s_task-mixedgamblestask_run-01_events.tsv'
    files = [join(path2, template % (s, s)) for s in subs]
    # Put them in a temporary directory
    tmp_dir = tempfile.mkdtemp()
    for f in files:
        shutil.copy2(f, tmp_dir)

    manager = load_variables([path, tmp_dir])
    col_keys = manager['time'].columns.keys()
    assert set(col_keys) == {'RT', 'gain', 'respnum', 'PTval', 'loss',
                             'respcat', 'parametric gain', 'trial_type'}
    col_keys = manager['subject'].columns.keys()
    print(manager['subject'].columns['type'].values)
    assert set(col_keys) == {'sex', 'age', 'type'}
    shutil.rmtree(tmp_dir)


def test_match_columns(manager):
    collection = manager['time']
    matches = collection.match_columns('^resp', return_type='columns')
    assert len(matches) == 2
    assert all(isinstance(m, SparseEventColumn) for m in matches)


def test_get_design_matrix(manager):
    sub_ids = [1, 2, 3, 4, 5, 6]
    subs = [str(s).zfill(2) for s in sub_ids]
    collection = manager['time']
    dm = collection.get_design_matrix(columns=['RT', 'parametric gain'],
                                      groupby=['subject', 'run'],
                                      subject=subs)
    assert set(dm['subject'].unique()) == set(sub_ids)
    cols = set(['amplitude', 'onset', 'duration', 'subject', 'run', 'task',
                'condition'])
    assert set(dm.columns) == cols
    assert set(dm['condition'].unique()) == {'RT', 'parametric gain'}
    assert dm.shape == (3072, 7)
