"""
Transformations that primarily involve manipulating/munging variables into
other formats or shapes.
"""

import numpy as np
import pandas as pd
# Might not be needed since python 3.7
# See https://github.com/bids-standard/pybids/issues/423 for more info
from collections import OrderedDict as odict

from bids.utils import listify
from .base import Transformation
from formulaic import model_matrix
import re
from bids.variables import DenseRunVariable, SimpleVariable


class Assign(Transformation):
    """Assign one variable's amplitude, duration, or onset attribute to
    another.
    """

    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'
    _allow_categorical = ('variables', 'target')

    def _transform(self, input, target, input_attr='amplitude',
                   target_attr='amplitude'):

        # assign only makes sense for sparse variables; dense variables don't
        # have durations or onsets, and amplitudes can be copied by cloning
        if isinstance(input, DenseRunVariable):
            raise ValueError("The 'assign' transformation can only be applied"
                             " to sparsely-coded event types. The input "
                             "variable (%s) is dense." % input.name)

        target = self.collection.variables[target].clone()
        if isinstance(target, DenseRunVariable):
            raise ValueError("The 'assign' transformation can only be applied"
                             " to sparsely-coded event types. The target "
                             "variable (%s) is dense." % target.name)

        # Ensure attributes are valid
        valid_attrs = ['amplitude', 'duration', 'onset']
        if input_attr not in valid_attrs:
            raise ValueError("Valid values for input_attr are: %s." %
                             valid_attrs)
        if target_attr not in valid_attrs:
            raise ValueError("Valid values for target_attr are: %s." %
                             valid_attrs)

        # variables must have same number of events, but do *not* need to have
        # aligned onsets.
        l_s, l_t = len(input.values), len(target.values)
        if l_s != l_t:
            raise ValueError("Input and target variables do not contain the "
                             "same number of events (%d vs. %d)." % (l_s, l_t))

        if input_attr.startswith('amplitude'):
            vals = input.values.values
        else:
            vals = getattr(input, input_attr)

        if target_attr.startswith('amplitude'):
            target.values[:] = vals
        else:
            setattr(target, target_attr, vals)

        return target


class Copy(Transformation):
    """Copy/clone a variable.

    Parameters
    ----------
    col : str
        Name of variable to copy.
    """

    _groupable = False
    _output_required = True
    _allow_categorical = ('variables',)

    def _transform(self, col):
        # We don't have to do anything else here b/c it's handled in base.
        return col


class Delete(Transformation):
    """Delete variables from the namespace.

    Parameters
    ----------
    variables : list or str
        Name(s) of variables to delete.
    """
    _groupable = False
    _loopable = False
    _input_type = 'variable'
    _return_type = 'none'
    _allow_categorical = ('variables',)

    def _transform(self, variables):
        variables = set([v.name for v in variables])
        self.collection.variables = {k: v for k, v in
                                     self.collection.variables.items()
                                     if k not in variables}


class DropNA(Transformation):

    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'
    _allow_categorical = ('variables',)

    def _transform(self, var):

        # Identify non-NA rows
        valid = var.values.notna().values
        var.select_rows(valid)
        return var


class Factor(Transformation):

    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'
    _allow_categorical = ('variables',)

    def _transform(self, var, constraint='none', ref_level=None, sep='.'):

        result = []
        data = var.to_df()
        orig_name = var.name
        variableClass = var.__class__

        levels = np.sort(data['amplitude'].unique())
        new_cols = pd.get_dummies(data['amplitude'], drop_first=False)[levels]

        if len(levels) > 1 and constraint in ('drop_one', 'mean_zero'):
            if ref_level is None:
                ref_level = levels[0]
            new_cols = new_cols.drop(ref_level, axis=1)

            if constraint == 'mean_zero':
                ref_inds = data['amplitude'] == ref_level
                new_cols.loc[ref_inds, :] = -1. / (len(levels) - 1)

        for lev in levels:
            if ref_level is not None and lev == ref_level:
                continue
            name = ''.join([var.name, sep, str(lev)])
            lev_data = data.copy()
            lev_data['amplitude'] = new_cols[lev].astype(float)
            args = [name, lev_data, var.source]
            if hasattr(var, 'run_info'):
                args.insert(2, var.run_info)
            new_col = variableClass(*args)
            result.append(new_col)

        # Remove existing variable. TODO: allow user to leave original in?
        self.collection.variables.pop(orig_name)

        return result


class Filter(Transformation):
    """Filter (remove) the values given a query to satisfy"""

    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'
    _aligned_required = 'force_dense'
    _aligned_variables = ('by')
    _allow_categorical = ('variables', 'by')

    def _transform(self, var, query, by=None):

        if by is None:
            by = []

        names = [var.name] + listify(by)

        # assure ordered dict so we have consistent (if not correct) operation,
        # because later we ask for name_map.values
        # pandas .query can't handle non-identifiers in variable names, so we
        # need to replace them in both the variable names and the query string.
        name_map = odict((n, re.sub('[^a-zA-Z0-9_]+', '_', n)) for n in names)
        for k, v in name_map.items():
            query = query.replace(k, v)

        data = pd.concat([self.collection[n].values for n in names],
                         axis=1, sort=True)
        # Make sure we can use integer index
        data = data.reset_index(drop=True)
        data.columns = list(name_map.values())
        data = data.query(query)

        # Truncate target variable to retained rows
        var.select_rows(data.index.values)

        return var


class Group(Transformation):
    """Groups a list of variables."""

    _groupable = False
    _loopable = False
    _input_type = 'variable'
    _return_type = 'none'

    def _transform(self, variables, name):
        if name in self.variables:
            raise ValueError("Variable group name '{}' conflicts with an "
                             "existing variable name!".format(name))
        self.collection.groups[name] = [v.name for v in variables]


class Rename(Transformation):
    """Rename a variable.

    Parameters
    ----------
    var : str
        Name of existing variable to rename.
    """
    _groupable = False
    _output_required = True
    _input_type = 'variable'
    _allow_categorical = ('variables',)

    def _transform(self, var):
        # Rename happens automatically in the base class, so all we need to
        # do is unset the original variable in the collection.
        self.collection.variables.pop(var.name)
        return var.values


class Replace(Transformation):
    """Replace values in the values, onset, or duration attributes."""
    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'
    _allow_categorical = ('variables',)

    def _transform(self, var, replace, attribute='value'):

        if attribute == 'value':
            var.values = var.values.replace(replace)
        elif attribute == 'onset':
            var.onset = pd.Series(var.onset).replace(replace).values
        elif attribute == 'duration':
            var.duration = pd.Series(var.duration).replace(replace).values
        else:
            raise ValueError("Invalid attribute. Must be one of 'value',"
                             "'onset', or 'duration'.")

        return var


class Select(Transformation):
    """Select variables to retain.

    Parameters
    ----------
    variables : list or str
        Name(s) of variables to retain. All variables
        not in the list will be dropped from the collection.
    """
    _groupable = False
    _loopable = False
    _input_type = 'variable'
    _return_type = 'none'
    _allow_categorical = ('variables',)

    def _transform(self, variables):
        self.collection.variables = {v.name: v for v in variables}


class Split(Transformation):
    """Split a single variable into N variables as defined by the levels of one
    or more other variables.

    Parameters
    ----------
    by : str or list
        Name(s) of variable(s) to split on.
    """

    _variables_used = ('variables', 'by')
    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'
    _allow_categorical = ('by',)
    _densify = ('variables', 'by')

    def _transform(self, var, by):

        if not isinstance(var, SimpleVariable):
            self._densify_variables()

        # Set up all the splitting variables as a DF. Note that variables in
        # 'by' can be either regular variables, or entities in the index--so
        # we need to check both places.
        all_variables = self._variables
        by_variables = [all_variables[v].values if v in all_variables
                        else var.index[v].reset_index(drop=True)
                        for v in listify(by)]
        group_data = pd.concat(by_variables, axis=1, sort=True)
        group_data.columns = listify(by)

        # Use formulaic to create splitting design matrix
        group_data = group_data.astype(str)
        formula = '0+' + ':'.join(listify(by))
        dm = model_matrix(formula, data=group_data)
        dm.columns = [col.replace(':', '.') for col in dm.columns]
        # formulaic output naming convention differs from patsy
        dm.columns = [col.replace('[T.', '[') for col in dm.columns]

        return var.split(dm)


class ToDense(Transformation):
    """Convert variable to dense representation."""

    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'

    def _transform(self, var, sampling_rate=10):
        return var.to_dense(sampling_rate=sampling_rate)


class Resample(Transformation):
    """ Frequency concersion and resampling of variable """
    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'

    def _transform(self, var, sampling_rate=10):
        return var.resample(sampling_rate=sampling_rate)
