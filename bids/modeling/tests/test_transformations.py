# from bids.modeling.variables import load_variables
from bids.modeling import transformations as transform
from bids.variables import SparseRunVariable
from bids.variables.entities import RunInfo
from bids.variables.collections import BIDSRunVariableCollection
from bids.layout import BIDSLayout
import math
import pytest
from os.path import join, sep
from bids.tests import get_test_data_path
import numpy as np
import pandas as pd

try:
    from unittest import mock
except ImportError:
    import mock


# Sub-select collection for faster testing, without sacrificing anything
# in the tests
SUBJECTS = ['01', '02']
NRUNS = 3
SCAN_LENGTH = 480

cached_collections = {}


@pytest.fixture
def collection():
    if 'ds005' not in cached_collections:
        layout_path = join(get_test_data_path(), 'ds005')
        layout = BIDSLayout(layout_path)
        cached_collections['ds005'] = layout.get_collections(
            'run',
            types=['events'],
            scan_length=SCAN_LENGTH,
            merge=True,
            sampling_rate=10,
            subject=SUBJECTS
        )
    # Always return a clone!
    return cached_collections['ds005'].clone()


@pytest.fixture
def sparse_run_variable_with_missing_values():
    data = pd.DataFrame({
        'onset': [2, 5, 11, 17],
        'duration': [1.2, 1.6, 0.8, 2],
        'amplitude': [1, 1, np.nan, 1]
    })
    run_info = [RunInfo({'subject': '01'}, 20, 2, 'dummy.nii.gz', 10)]
    var = SparseRunVariable(
        name='var', data=data, run_info=run_info, source='events')
    return BIDSRunVariableCollection([var])


def test_convolve(collection):
    rt = collection.variables['RT']
    transform.Convolve(collection, 'RT', output='reaction_time')
    rt_conv = collection.variables['reaction_time']

    assert rt_conv.values.shape[0] == \
        rt.get_duration() * collection.sampling_rate

    transform.ToDense(collection, 'RT', output='rt_dense')
    transform.Convolve(collection, 'rt_dense', output='dense_convolved')

    dense_conv = collection.variables['reaction_time']

    assert dense_conv.values.shape[0] == \
        rt.get_duration() * collection.sampling_rate

    # Test adapative oversampling computation
    # Events are 3s duration events every 4s, so resolution demanded by the data is 1Hz
    # To resolve 1Hz frequencies, we must sample at >=2Hz
    args = (mock.ANY, 'spm', mock.ANY)
    kwargs = dict(fir_delays=None, min_onset=0)
    with mock.patch('bids.modeling.transformations.compute.hrf') as mocked:
        # Sampling rate is 10Hz, no oversampling needed
        transform.Convolve(collection, 'RT', output='rt_mock')
        mocked.compute_regressor.assert_called_with(*args, oversampling=1.0, **kwargs)

    with mock.patch('bids.modeling.transformations.compute.hrf') as mocked:
        # Sampling rate is 10Hz, no oversampling needed
        transform.Convolve(collection, 'rt_dense', output='rt_mock')
        mocked.compute_regressor.assert_called_with(*args, oversampling=1.0, **kwargs)

    with mock.patch('bids.modeling.transformations.compute.hrf') as mocked:
        # Slow sampling rate, oversample (4x) to 2Hz
        collection.sampling_rate = 0.5
        transform.Convolve(collection, 'RT', output='rt_mock')
        mocked.compute_regressor.assert_called_with(*args, oversampling=4.0, **kwargs)

    with mock.patch('bids.modeling.transformations.compute.hrf') as mocked:
        # Dense variable is already sampled at 10Hz, no oversampling needed
        collection.sampling_rate = 0.5
        transform.Convolve(collection, 'rt_dense', output='rt_mock')
        mocked.compute_regressor.assert_called_with(*args, oversampling=1.0, **kwargs)

    with mock.patch('bids.modeling.transformations.compute.hrf') as mocked:
        # Onset requires 10Hz resolution, oversample (2x) to 20Hz
        collection.sampling_rate = 10
        collection['RT'].onset[0] += 0.1
        transform.Convolve(collection, 'RT', output='rt_mock')
        mocked.compute_regressor.assert_called_with(*args, oversampling=2.0, **kwargs)


def test_convolve_impulse():
    # Smoke test impulse convolution
    data = pd.DataFrame({
        'onset': [10, 20],
        'duration': [0, 0],
        'amplitude': [1, 1]
    })
    run_info = [RunInfo({'subject': '01'}, 20, 2, 'dummy.nii.gz', 10)]
    var = SparseRunVariable(
        name='var', data=data, run_info=run_info, source='events')
    coll = BIDSRunVariableCollection([var])
    transform.ToDense(coll, 'var', output='var_dense')
    transform.Convolve(coll, 'var', output='var_hrf')
    transform.Convolve(coll, 'var_dense', output='var_dense_hrf')


def test_rename(collection):
    dense_rt = collection.variables['RT'].to_dense(collection.sampling_rate)
    assert len(dense_rt.values) == math.ceil(len(SUBJECTS) * NRUNS * SCAN_LENGTH * collection.sampling_rate)
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
    transform.Sum(
        collection, variables=['parametric gain', 'gain'], output='sum')
    res = c['sum'].values
    target = c['parametric gain'].values + c['gain'].values
    assert np.array_equal(res, target)
    transform.Sum(
        collection,
        variables=['parametric gain', 'gain'], output='sum', weights=[2, 2])
    assert np.array_equal(c['sum'].values, target * 2)
    with pytest.raises(ValueError):
        transform.Sum(collection, variables=['parametric gain', 'gain'],
                      output='sum', weights=[1, 1, 1])


def test_scale(collection, sparse_run_variable_with_missing_values):
    transform.Scale(collection, variables=['RT', 'parametric gain'],
                    output=['RT_Z', 'gain_Z'], groupby=['run', 'subject'])
    groupby = collection['RT'].get_grouper(['run', 'subject'])
    z1 = collection['RT_Z'].values
    z2 = collection['RT'].values.groupby(
        groupby).apply(lambda x: (x - x.mean()) / x.std())
    assert np.allclose(z1, z2)

    # Test constant input
    coll = sparse_run_variable_with_missing_values
    coll['var'].values.fillna(1)
    with pytest.raises(ValueError, match="Cannot scale.*1.0"):
        transform.Scale(coll, 'var').values
    with pytest.raises(ValueError, match="Cannot scale.*1.0"):
        transform.Scale(coll, 'var', replace_na='before').values
    transform.Scale(coll, 'var', replace_na='after', output='zero')
    assert coll['zero'].values.unique() == 0


def test_demean(collection):
    transform.Demean(collection, variables=['RT'], output=['RT_dm'])
    m1 = collection['RT_dm'].values
    m2 = collection['RT'].values
    m2 -= m2.values.mean()
    assert np.allclose(m1, m2)


def test_orthogonalize_dense(collection):
    transform.Factor(collection, 'trial_type', sep=sep)

    sampling_rate = collection.sampling_rate
    # Store pre-orth variables needed for tests
    pg_pre = collection['trial_type/parametric gain'].to_dense(sampling_rate)
    rt = collection['RT'].to_dense(sampling_rate)

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
    collection['RT_3'] = collection['RT']\
        .clone(name='RT_3').to_dense(collection.sampling_rate)

    rt_pre_onsets = collection['RT'].onset
    rt_pre_values = collection['RT'].values.values

    # Grouping SparseEventVariable by one column
    transform.Split(collection, ['RT'], ['respcat'])

    # Verify names
    assert 'RT.respcat[0]' in collection.variables.keys() and \
           'RT.respcat[-1]' in collection.variables.keys()

    # Verify values
    rt_post_onsets = np.r_[collection['RT.respcat[0]'].onset,
                           collection['RT.respcat[-1]'].onset,
                           collection['RT.respcat[1]'].onset]
    assert np.array_equal(rt_pre_onsets.sort(), rt_post_onsets.sort())

    rt_post_values = np.r_[collection['RT.respcat[0]'].values.values,
                           collection['RT.respcat[-1]'].values.values,
                           collection['RT.respcat[1]'].values.values]
    assert np.array_equal(rt_pre_values.sort(), rt_post_values.sort())

    # Grouping SparseEventVariable by multiple columns
    transform.Split(collection, variables=['RT_2'], by=['respcat', 'loss'])
    assert 'RT_2.loss[13].respcat[-1]' in collection.variables.keys() and \
           'RT_2.loss[13].respcat[1]' in collection.variables.keys()

    # Grouping by DenseEventVariable
    transform.Split(collection, variables='RT_3', by='respcat')
    targets = ['RT_3.respcat[-1]', 'RT_3.respcat[0]', 'RT_3.respcat[1]']
    assert not set(targets) - set(collection.variables.keys())
    assert collection['respcat'].values.nunique() == 3
    n_dense = len(collection['RT_3'].values)
    assert len(collection['RT_3.respcat[-1]'].values) == n_dense

    # Grouping by entities in the index
    collection['RT_4'] = orig.clone(name='RT_4')
    transform.Split(collection, variables=['RT_4'], by=['respcat', 'run'])
    assert 'RT_4.respcat[-1].run[3]' in collection.variables.keys()


def test_resample_dense(collection):
    new_sampling_rate = 50
    old_sampling_rate = collection.sampling_rate
    upsampling = float(new_sampling_rate) / old_sampling_rate

    collection['RT'] = collection['RT'].to_dense(old_sampling_rate)
    old_rt = collection['RT'].clone()
    collection.resample(new_sampling_rate, in_place=True)
    assert math.floor(len(old_rt.values) * upsampling) == len(collection['RT'].values)
    # Should work after explicitly converting categoricals
    transform.Factor(collection, 'trial_type')

    new_sampling_rate2 = 5
    upsampling2 = float(new_sampling_rate2) / old_sampling_rate

    collection.resample(new_sampling_rate2, force_dense=True, in_place=True)
    assert len(old_rt.values) == math.ceil(float(len(collection['parametric gain'].values) / upsampling2))


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


def test_expand_variable_names(collection):
    # Should fail because two output values are required following expansion
    with pytest.raises(Exception):
        transform.Copy(collection, '*resp*')

    transform.Copy(collection, '*resp*', output_suffix='_copy')
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
    assert len(collection['RT'].values) == 96 * len(SUBJECTS)


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

    coll.variables['respnum.0'].onset += 1

    # Should fail because I misaligned variable
    with pytest.raises(ValueError):
        transform.And(coll, names, output='misaligned')

    # Should pass because dense is set to True and will align
    transform.And(coll, names, output='misaligned', dense=True)


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


def test_group(collection):
    coll = collection.clone()
    with pytest.raises(ValueError):
        # Can't use an existing variable name as the group name
        transform.Group(coll, ['gain', 'loss'], name='gain')

    transform.Group(coll, ['gain', 'loss'], name='outcome_vars')
    assert coll.groups == {'outcome_vars': ['gain', 'loss']}

    # Checks that variable groups are replaced properly
    transform.Rename(coll, ['outcome_vars'],
                     output=['gain_renamed', 'loss_renamed'])
    assert 'gain_renamed' in coll.variables
    assert 'loss_renamed' in coll.variables
    assert 'gain' not in coll.variables
    assert 'loss' not in coll.variables


def test_resample(collection):
    coll = collection.clone()

    transform.ToDense(coll, 'parametric gain', output='pg_dense')
    pg = coll.variables['pg_dense']
    old_shape = pg.values.shape
    old_auc = np.trapz(np.abs(pg.values.values.squeeze()), dx=0.1)
    transform.Resample(coll, 'pg_dense', 1)
    pg = coll.variables['pg_dense']
    new_shape = pg.values.shape
    # Spacing (dx) is 10* larger when downsampled fro 10hz to 1hz
    new_auc = np.trapz(np.abs(pg.values.values.squeeze()), dx=1)

    # Shape from 10hz to 1hz
    assert new_shape[0] == old_shape[0] / 10

    # Assert that the auc is more or less the same (not exact, rounding error)
    # Values are around 0.25
    assert np.allclose(old_auc, new_auc, rtol=0.05)
