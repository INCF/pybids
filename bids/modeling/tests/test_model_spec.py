from os.path import join
import json

import pandas as pd
import numpy as np
import pytest

from bids.variables import BIDSVariableCollection
from bids.modeling.model_spec import (GLMMSpec, Term, VarComp)
from bids.modeling import TransformerManager
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path


@pytest.fixture(scope='module')
def collection():
    """Create a collection at dataset level."""
    df = pd.DataFrame({
        'subject': np.repeat(np.arange(10), 4),
        'run': np.tile(np.arange(4), 10),
        'age': np.repeat(np.random.normal(40, 10, 10).astype(int), 4)
    })
    return BIDSVariableCollection.from_df(df[['age']], entities=df[['subject', 'run']])


def test_df_to_model_spec(collection):
    layout_path = join(get_test_data_path(), 'ds005')
    json_file = join(layout_path, 'models', 'ds-005_type-mfx_model.json')
    spec = json.load(open(json_file, 'r'))
    df = collection.to_df()
    data = df[['age']]
    metadata = df[['subject', 'run']]
    md = GLMMSpec.from_df(data, spec['Nodes'][1]['Model'], metadata)

    assert len(md.terms) == 2
    assert md.terms['age'].values.shape == (40, )
    assert md.terms['SubjectSlopes'].values.shape == (40, 10)
    index_vec = md.terms['SubjectSlopes'].index_vec
    assert index_vec.shape == (40,)
    assert np.array_equal(np.sort(np.unique(index_vec)), np.arange(10) + 1)
    assert md.Z.shape == (40, 10)
    assert md.Z.columns[0] == 'SubjectSlopes.0'


def test_glmmspec_empty_init():
    md = GLMMSpec()
    assert md.terms == {}
    assert md.X is None
    assert md.Z is None
    assert md.family is None
    assert md.family is None
    assert md.sigma is None


def test_fixed_term_init():
    t = Term('dummy', np.random.normal(size=20), categorical=True)
    assert t.name == 'dummy'
    assert t.values.shape == (20,)
    assert t.categorical


def test_var_comp_init():
    Z = np.repeat(np.eye(20), 5, axis=0)
    t = VarComp('sigma', Z)
    assert t.name == 'sigma'
    assert t.values.shape == (100, 20)
    assert t.categorical
