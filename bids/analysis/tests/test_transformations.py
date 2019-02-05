# from bids.analysis.variables import load_variables
from bids.analysis import transformations as transform
from bids.variables import SparseRunVariable
from bids.variables.entities import RunInfo
from bids.variables.kollekshuns import BIDSRunVariableCollection
from bids.layout import BIDSLayout
import pytest
from os.path import join, sep
from bids.tests import get_test_data_path
import numpy as np
import pandas as pd


@pytest.fixture
def collection():
    layout_path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(layout_path)
    collection = layout.get_collections('run', types=['events'],
                                        scan_length=480, merge=True,
                                        sampling_rate=10)
    return collection


@pytest.fixture
def sparse_run_variable_with_missing_values():
    data = pd.DataFrame({
        'onset': [2, 5, 11, 17],
        'duration': [1.2, 1.6, 0.8, 2],
        'amplitude': [1, 1, np.nan, 1]
    })
    run_info = [RunInfo({'subject': '01'}, 20, 2, 2, 'dummy.nii.gz')]
    var = SparseRunVariable('var', data, run_info, 'events')
    return BIDSRunVariableCollection([var])


def test_rename(collection):
    dense_rt = collection.variables['RT'].to_dense(10)
    assert len(dense_rt.values) == 230400
    transform.Rename(collection, 'RT', output='reaction_time')
    assert 'reaction_time' in collection.variables
    assert 'RT' not in collection.variables
    col = collection.variables['reaction_time']
    assert col.name == 'reaction_time'
    assert col.onset.max() == 476


def test_product(collection):
    c = collection
    transform.Product(collection, variables=['parametric gain', 'gain'],
                      output='prod')
    res = c['prod'].values
    assert (res == c['parametric gain'].values * c['gain'].values).all()


def test_sum(collection):
    c = collection
    transform.Sum(collection, variables=['parametric gain', 'gain'],
                      output='sum')
    res = c['sum'].values
    target = c['parametric gain'].values + c['gain'].values
    assert np.array_equal(res, target)
    transform.Sum(collection, variables=['parametric gain', 'gain'],
                      output='sum', weights=[2, 2])
    assert np.array_equal(c['sum'].values, target * 2)
    with pytest.raises(ValueError):
        transform.Sum(collection, variables=['parametric gain', 'gain'],
                      output='sum', weights=[1, 1, 1])


def test_scale(collection):
    transform.Scale(collection, variables=['RT', 'parametric gain'],
                    output=['RT_Z', 'gain_Z'], groupby=['run', 'subject'])
    groupby = collection['RT'].get_grouper(['run', 'subject'])
    z1 = collection['RT_Z'].values
    z2 = collection['RT'].values.groupby(
        groupby).apply(lambda x: (x - x.mean()) / x.std())
    assert np.allclose(z1, z2)


def test_demean(collection):
    transform.Demean(collection, variables=['RT'], output=['RT_dm'])
    m1 = collection['RT_dm'].values
    m2 = collection['RT'].values
    m2 -= m2.values.mean()
    assert np.allclose(m1, m2)


def test_orthogonalize_dense(collection):
    transform.Factor(collection, 'trial_type', sep=sep)

    # Store pre-orth variables needed for tests
    pg_pre = collection['trial_type/parametric gain'].to_dense(10)
    rt = collection['RT'].to_dense(10)

    # Orthogonalize and store result
    transform.Orthogonalize(collection, variables='trial_type/parametric gain',
                            other='RT', dense=True, groupby=['run', 'subject'])
    pg_post = collection['trial_type/parametric gain']

    # Verify that the to_dense() calls result in identical indexing
    ent_cols = ['subject', 'run']
    assert pg_pre.to_df()[ent_cols].equals(rt.to_df()[ent_cols])
    assert pg_post.to_df()[ent_cols].equals(rt.to_df()[ent_cols])

    vals = np.c_[rt.values, pg_pre.values, pg_post.values]
    df = pd.DataFrame(vals, columns=['rt', 'pre', 'post'])
    groupby = rt.get_grouper(['run', 'subject'])
    pre_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 1])
    post_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 2])
    assert (pre_r > 0.2).any()
    assert (post_r < 0.0001).all()


def test_orthogonalize_sparse(collection):
    pg_pre = collection['parametric gain'].values
    rt = collection['RT'].values
    transform.Orthogonalize(collection, variables='parametric gain',
                            other='RT', groupby=['run', 'subject'])
    pg_post = collection['parametric gain'].values
    vals = np.c_[rt.values, pg_pre.values, pg_post.values]
    df = pd.DataFrame(vals, columns=['rt', 'pre', 'post'])
    groupby = collection['RT'].get_grouper(['run', 'subject'])
    pre_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 1])
    post_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 2])
    assert (pre_r > 0.2).any()
    assert (post_r < 0.0001).all()


def test_split(collection):

    orig = collection['RT'].clone(name='RT_2')
    collection['RT_2'] = orig.clone()
    collection['RT_3'] = collection['RT'].clone(name='RT_3').to_dense(10)

    rt_pre_onsets = collection['RT'].onset

    # Grouping SparseEventVariable by one column
    transform.Split(collection, ['RT'], ['respcat'])
    assert 'RT.0' in collection.variables.keys() and \
           'RT.-1' in collection.variables.keys()
    rt_post_onsets = np.r_[collection['RT.0'].onset,
                           collection['RT.-1'].onset,
                           collection['RT.1'].onset]
    assert np.array_equal(rt_pre_onsets.sort(), rt_post_onsets.sort())

    # Grouping SparseEventVariable by multiple columns
    transform.Split(collection, variables=['RT_2'], by=['respcat', 'loss'])
    assert 'RT_2.-1_13' in collection.variables.keys() and \
           'RT_2.1_13' in collection.variables.keys()

    # Grouping by DenseEventVariable
    transform.Split(collection, variables='RT_3', by='respcat',
                    drop_orig=False)
    targets = ['RT_3.respcat[-1]', 'RT_3.respcat[0]', 'RT_3.respcat[1]']
    assert not set(targets) - set(collection.variables.keys())
    assert collection['respcat'].values.nunique() == 3
    n_dense = len(collection['RT_3'].values)
    assert len(collection['RT_3.respcat[-1]'].values) == n_dense

    # Grouping by entities in the index
    collection['RT_4'] = orig.clone(name='RT_4')
    transform.Split(collection, variables=['RT_4'], by=['respcat', 'run'])
    assert 'RT_4.-1_3' in collection.variables.keys()


def test_resample_dense(collection):
    collection['RT'] = collection['RT'].to_dense(10)
    old_rt = collection['RT'].clone()
    collection.resample(50, in_place=True)
    assert len(old_rt.values) * 5 == len(collection['RT'].values)
    # Should work after explicitly converting categoricals
    transform.Factor(collection, 'trial_type')
    collection.resample(5, force_dense=True, in_place=True)
    assert len(old_rt.values) == len(collection['parametric gain'].values) * 2


def test_threshold(collection):
    old_pg = collection['parametric gain']
    orig_vals = old_pg.values

    collection['pg'] = old_pg.clone()
    transform.Threshold(collection, 'pg', threshold=0.2, binarize=True)
    assert collection.variables['pg'].values.sum() == (orig_vals >= 0.2).sum()

    collection['pg'] = old_pg.clone()
    transform.Threshold(collection, 'pg', threshold=0.2, binarize=False)
    assert collection.variables['pg'].values.sum() != (orig_vals >= 0.2).sum()
    coll_sum = (collection.variables['pg'].values >= 0.2).sum()
    assert coll_sum == (orig_vals >= 0.2).sum()

    collection['pg'] = old_pg.clone()
    transform.Threshold(collection, 'pg', threshold=-0.1, binarize=True,
                        signed=False, above=False)
    n = np.logical_and(orig_vals <= 0.1, orig_vals >= -0.1).sum()
    assert collection.variables['pg'].values.sum() == n


def test_assign(collection):
    transform.Assign(collection, 'parametric gain', target='RT',
                     target_attr='onset', output='test1')
    t1 = collection['test1']
    pg = collection['parametric gain']
    rt = collection['RT']
    assert np.array_equal(t1.onset, pg.values.values)
    assert np.array_equal(t1.duration, rt.duration)
    assert np.array_equal(t1.values.values, rt.values.values)

    transform.Assign(collection, 'RT', target='parametric gain',
                     input_attr='onset', target_attr='amplitude',
                     output='test2')
    t2 = collection['test2']
    assert np.array_equal(t2.values.values, rt.onset)
    assert np.array_equal(t2.onset, pg.onset)
    assert np.array_equal(t2.duration, pg.duration)


def test_copy(collection):
    transform.Copy(collection, 'RT', output='RT_copy')
    assert 'RT_copy' in collection.variables.keys()
    assert np.array_equal(collection['RT'].values.values,
                          collection['RT_copy'].values.values)


def test_regex_variable_expansion(collection):
    # Should fail because two output values are required following expansion
    with pytest.raises(Exception):
        transform.Copy(collection, 'resp', regex_variables='variables')

    transform.Copy(collection, 'resp', output_suffix='_copy',
                   regex_variables='variables')
    assert 'respnum_copy' in collection.variables.keys()
    assert 'respcat_copy' in collection.variables.keys()
    assert np.array_equal(collection['respcat'].values.values,
                          collection['respcat_copy'].values.values)
    assert np.array_equal(collection['respnum'].values.values,
                          collection['respnum_copy'].values.values)


def test_factor(collection):
    # Full-rank dummy-coding, only one unique value in variable
    trial_type = collection.variables['trial_type'].clone()
    coll = collection.clone()
    transform.Factor(coll, 'trial_type', sep='@')
    assert 'trial_type@parametric gain' in coll.variables.keys()
    pg = coll.variables['trial_type@parametric gain']
    assert pg.values.unique() == [1]
    assert pg.values.shape == trial_type.values.shape

    # Reduced-rank dummy-coding, only one unique value in variable
    coll = collection.clone()
    transform.Factor(coll, 'trial_type', constraint='mean_zero')
    assert 'trial_type.parametric gain' in coll.variables.keys()
    pg = coll.variables['trial_type.parametric gain']
    assert pg.values.unique() == [1]
    assert pg.values.shape == trial_type.values.shape

    # full-rank dummy-coding, multiple values
    coll = collection.clone()
    transform.Factor(coll, 'respnum')
    targets = set(['respnum.%d' % d for d in range(0, 5)])
    assert not targets - set(coll.variables.keys())
    assert all([set(coll.variables[t].values.unique()) == {0.0, 1.0}
                for t in targets])
    data = pd.concat([coll.variables[t].values for t in targets],
                     axis=1, sort=True)
    assert (data.sum(1) == 1).all()

    # reduced-rank dummy-coding, multiple values
    coll = collection.clone()
    transform.Factor(coll, 'respnum', constraint='drop_one')
    targets = set(['respnum.%d' % d for d in range(1, 5)])
    assert not targets - set(coll.variables.keys())
    assert 'respnum.0' not in coll.variables.keys()
    assert all([set(coll.variables[t].values.unique()) == {0.0, 1.0}
                for t in targets])
    data = pd.concat([coll.variables[t].values for t in targets],
                     axis=1, sort=True)
    assert set(np.unique(data.sum(1).values.ravel())) == {0., 1.}

    # Effect coding, multiple values
    coll = collection.clone()
    transform.Factor(coll, 'respnum', constraint='mean_zero')
    targets = set(['respnum.%d' % d for d in range(1, 5)])
    assert not targets - set(coll.variables.keys())
    assert 'respnum.0' not in coll.variables.keys()
    assert all([set(coll.variables[t].values.unique()) == {-0.25, 0.0, 1.0}
                for t in targets])
    data = pd.concat([coll.variables[t].values for t in targets],
                     axis=1, sort=True)
    assert set(np.unique(data.sum(1).values.ravel())) == {-1., 1.}


def test_filter(collection):
    orig = collection['parametric gain'].clone()
    q = "parametric gain > 0"
    transform.Filter(collection, 'parametric gain', query=q)
    assert len(orig.values) == 2 * len(collection['parametric gain'].values)
    assert np.all(collection['parametric gain'].values > 0)

    orig = collection['RT'].clone()
    q = 'parametric gain > 0.1'
    transform.Filter(collection, 'RT', query=q, by='parametric gain')
    assert len(orig.values) != len(collection['RT'].values)
    # There is some bizarro thing going on where, on travis, the result is
    # randomly either 1536 or 3909 when running on Python 3 (on linux or mac).
    # Never happens locally, and I've had no luck tracking down the problem.
    # Best guess is it reflects either some non-deterministic ordering of
    # variables somewhere, or some weird precision issues when resampling to
    # dense. Needs to be tracked down and fixed.
    assert len(collection['RT'].values) in [1536, 3909]


def test_replace(collection):
    orig = collection['parametric gain'].clone()
    # Values
    replace_dict = {0.0335: 2.0, -0.139: 2.0}
    transform.Replace(collection, 'parametric gain', replace_dict)
    target = set(orig.values.unique()) - {0.0335, -0.139} | {2.0}
    assert set(collection['parametric gain'].values.unique()) == target
    # Durations
    replace_dict = {3: 2}
    transform.Replace(collection, 'parametric gain', replace_dict, 'duration')
    target = set(np.unique(orig.duration)) - {3} | {2.0}
    assert set(np.unique(collection['parametric gain'].duration)) == target
    # Onsets
    replace_dict = {4.: 3., 476.: 475.5}
    transform.Replace(collection, 'parametric gain', replace_dict, 'onset')
    target = set(np.unique(orig.onset)) - {4., 476.} | {3., 475.5}
    assert set(np.unique(collection['parametric gain'].onset)) == target


def test_select(collection):
    coll = collection.clone()
    keep = ['RT', 'parametric gain', 'respcat']
    transform.Select(coll, keep)
    assert set(coll.variables.keys()) == set(keep)


def test_delete(collection):
    coll = collection.clone()
    all_cols = set(coll.variables.keys())
    drop = ['RT', 'parametric gain', 'respcat']
    transform.Delete(coll, drop)
    assert all_cols - set(coll.variables.keys()) == set(drop)


def test_and(collection):
    coll = collection.clone()
    transform.Factor(coll, 'respnum')
    names = ['respnum.%d' % d for d in range(0, 5)]
    transform.And(coll, names, output='conjunction')
    assert not coll.variables['conjunction'].values.sum()

    coll['copy'] = coll.variables['respnum.0'].clone()
    transform.And(coll, ['respnum.0', 'copy'], output='conj')
    assert coll.variables['conj'].values.astype(float).equals(
        coll.variables['respnum.0'].values)


def test_or(collection):
    coll = collection.clone()
    transform.Factor(coll, 'respnum')
    names = ['respnum.%d' % d for d in range(0, 5)]
    transform.Or(coll, names, output='disjunction')
    assert (coll.variables['disjunction'].values == 1).all()

    coll['copy'] = coll.variables['respnum.0'].clone()
    transform.Or(coll, ['respnum.0', 'copy'], output='or')
    assert coll.variables['or'].values.astype(float).equals(
        coll.variables['respnum.0'].values)


def test_not(collection):
    coll = collection.clone()
    pre_rt = coll.variables['RT'].values.values
    transform.Not(coll, 'RT')
    post_rt = coll.variables['RT'].values.values
    assert (post_rt == ~pre_rt.astype(bool)).all()


def test_dropna(sparse_run_variable_with_missing_values):
    var = sparse_run_variable_with_missing_values.variables['var']
    coll = sparse_run_variable_with_missing_values.clone()
    transform.DropNA(coll, 'var')
    post_trans = coll.variables['var']
    assert len(var.values) > len(post_trans.values)
    assert np.array_equal(post_trans.values, [1, 1, 1])
    assert np.array_equal(post_trans.onset, [2, 5, 17])
    assert np.array_equal(post_trans.duration, [1.2, 1.6, 2])
    assert len(post_trans.index) == 3
