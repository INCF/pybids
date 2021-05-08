""" Classes for representing BIDS variables. """

import math
import warnings
from copy import deepcopy
from abc import abstractmethod, ABCMeta
from itertools import chain
from functools import reduce

import numpy as np
import pandas as pd

from bids.utils import listify
from bids.utils import matches_entities

class BIDSVariable(metaclass=ABCMeta):
    """Base representation of a column in a BIDS project. """

    # Columns that define special properties (e.g., onset, duration). These
    # will be stored separately from the main data object, and are accessible
    # as properties on the BIDSVariable instance.
    _property_columns = set()

    def __init__(self, name, values, source):
        self.name = name
        self.values = values
        self.source = source
        self.entities = self._extract_entities()

    def clone(self, data=None, **kwargs):
        """Clone (deep copy) the current column, optionally replacing its
        data and/or any other attributes.

        Parameters
        ----------
        data : :obj:`pandas.DataFrame` or array_like
            Optional new data to substitute into
            the cloned column. Must have same dimensionality as the
            original.
        kwargs : dict
            Optional keyword arguments containing new attribute
            values to set in the copy. E.g., passing `name='my_name'`
            would set the `.name` attribute on the cloned instance to the
            passed value.
        """
        result = deepcopy(self)
        if data is not None:
            if data.shape != self.values.shape:
                # If data can be re-shaped safely, do so
                if data.squeeze().shape == self.values.squeeze().shape:
                    data = data.values.reshape(self.values.shape)
                else:
                    raise ValueError("Replacement data has shape %s; must have"
                                     " same shape as existing data %s." %
                                     (data.shape, self.values.shape))
            result.values = pd.DataFrame(data)

        if kwargs:
            for k, v in kwargs.items():
                setattr(result, k, v)

        # Need to update name on Series as well
        # result.values.name = kwargs.get('name', self.name)
        return result

    def filter(self, filters=None, query=None, strict=False, inplace=False):
        """Returns a copy of the current Variable with only rows that match
        the filters retained.

        Parameters
        ----------
        filters : dict
            Dictionary of filters to apply. Keys can be either
            'amplitude' or any named entity. Values must be single values
            or lists.
        query : str
            Optional query string to pass to df.query(). Will not
            be validated in any way, so must have valid column names. Takes
            precedence over filters in the event that both are passed.
        strict : bool
            By default, keys in 'filters' that cannot be found
            in the Variable will be silently ignored. If strict=True, None
            will be returned in such cases.
        inplace : bool
            If True, filtering is performed in place. If False,
            a filtered copy of the Variable is returned.

        Returns
        -------
        BIDSVariable or None if no rows are left after filtering.
        """

        if filters is None and query is None:
            raise ValueError("Either the 'filters' or the 'query' argument "
                             "must be provided!")

        if filters is not None and query is None:
            query = []
            for name, val in filters.items():
                if name != 'amplitude' and name not in self.index.columns:
                    if strict:
                        return None
                    continue
                oper = 'in' if isinstance(val, (list, tuple)) else '=='
                q = '{name} {oper} {val}'.format(name=name, oper=oper,
                                                 val=repr(val))
                query.append(q)
            query = ' and '.join(query)

        var = self if inplace else self.clone()

        if query:
            inds = self.to_df().query(query).index

            var.values = var.values.loc[inds]
            var.index = var.index.loc[inds]
            if hasattr(self, '_build_entity_index'):
                var._build_entity_index()

        if not inplace:
            return var

    @classmethod
    def merge(cls, variables, name=None, **kwargs):
        """Merge/concatenate a list of variables along the row axis.

        Parameters
        ----------
        variables : list
            A list of Variables to merge.
        name : str
            Optional name to assign to the output Variable. By
            default, uses the same name as the input variables.
        kwargs : dict
            Optional keyword arguments to pass onto the class-specific
            merge() call. See merge_variables docstring for details.

        Returns
        -------
        A single BIDSVariable of the same class as the input variables.

        See also
        --------
        merge_variables
        """

        variables = listify(variables)
        if len(variables) == 1:
            return variables[0]

        var_names = set([v.name for v in variables])
        if len(var_names) > 1:
            raise ValueError("Columns with different names cannot be merged. "
                             "Column names provided: %s" % var_names)

        if name is None:
            name = variables[0].name

        return cls._merge(variables, name, **kwargs)

    @classmethod
    @abstractmethod
    def _merge(cls, variables, name, **kwargs):
        pass

    def get_grouper(self, groupby='run'):
        """Return a list suitable for use in groupby calls.

        Parameters
        ----------
        groupby : str or list
            Name(s) of column(s) defining the grouper
            object. Anything that would be valid inside a .groupby() call
            on a pandas structure.

        Returns
        -------
        list
            A list defining the groups.
        """
        grouper = self.index.loc[:, groupby]
        return grouper.apply(lambda x: '@@@'.join(x.astype(str).values),
                             axis=1)

    def apply(self, func, groupby='run', *args, **kwargs):
        """Applies the passed function to the groups defined by the groupby
        argument. Works identically to the standard pandas df.groupby() call.

        Parameters
        ----------
        func : callable
            The function to apply to each group.
        groupby : str or list
            Name(s) of column(s) defining the grouping.
        args, kwargs : dict
            Optional positional and keyword arguments to pass
            onto the function call.
        """
        grouper = self.get_grouper(groupby)
        return self.values.groupby(grouper).apply(func, *args, **kwargs)

    def to_df(self, condition=True, entities=True, **kwargs):
        """Convert to a DataFrame, with columns for name and entities.

        Parameters
        ----------
        condition : bool
            If True, adds a column for condition name, and
            names the amplitude column 'amplitude'. If False, returns just
            onset, duration, and amplitude, and gives the amplitude column
            the current column name.
        entities : bool
            If True, adds extra columns for all entities.
        """
        amp = 'amplitude' if condition else self.name
        data = pd.DataFrame({amp: self.values.values.ravel()})

        for sc in self._property_columns:
            data[sc] = getattr(self, sc)

        if condition:
            data['condition'] = self.name

        if entities:
            ent_data = self.index.reset_index(drop=True)
            data = pd.concat([data, ent_data], axis=1, sort=True)

        return data.reset_index(drop=True)

    def _extract_entities(self):
        """Returns a dict of all non-varying entities for the current Variable.

        Notes
        -----
        Only entity key/value pairs common to all rows in the Variable
        are returned. E.g., if a Variable contains events extracted from
        runs 1, 2 and 3 from subject '01', the returned dict will be
        {'subject': '01'}; the runs will be excluded as they vary across
        the Variable contents.
        """
        constant = self.index.apply(lambda x: x.nunique() == 1)
        if constant.empty:
            return {}
        else:
            keep = self.index.columns[constant]
            return {k: self.index[k].dropna().iloc[0] for k in keep}


class SimpleVariable(BIDSVariable):
    """Represents a simple design matrix column that has no timing
    information.

    Parameters
    ----------
    name : str
        Name of the column.
    data : :obj:`pandas.DataFrame`
        A pandas DataFrame minimally containing a column
        named 'amplitude' as well as any identifying entities.
    source : str
        The type of BIDS variable file the data were extracted
        from. Must be one of: 'events', 'physio', 'stim', 'regressors',
        'scans', 'sessions', 'participants', or 'beh'.
    kwargs : dict
        Optional keyword arguments passed onto superclass.
    """

    _entity_columns = {'condition', 'amplitude'}

    def __init__(self, name, data, source, **kwargs):

        ent_cols = list(set(data.columns) - self._entity_columns)
        self.index = data.loc[:, ent_cols]

        values = data['amplitude'].reset_index(drop=True)
        values.name = name

        super(SimpleVariable, self).__init__(name, values, source)

    def split(self, grouper):
        """Split the current SparseRunVariable into multiple columns.

        Parameters
        ----------
        grouper : :obj:`pandas.DataFrame`
            Binary DF specifying the design matrix to use for splitting. Number
            of rows must match current ``SparseRunVariable``;
            a new ``SparseRunVariable`` will be generated for each column in
            the grouper.

        Returns
        -------
        A list of SparseRunVariables, one per column in the grouper DF.
        """
        data = self.to_df(condition=True, entities=True)
        data = data.drop('condition', axis=1)

        subsets = []
        for i, col_name in enumerate(grouper.columns):
            col_data = data.loc[grouper[col_name].astype(bool), :]
            name = '{}.{}'.format(self.name, col_name)
            col = self.__class__(name=name, data=col_data, source=self.source,
                                 run_info=getattr(self, 'run_info', None))
            subsets.append(col)
        return subsets

    @classmethod
    def _merge(cls, variables, name, **kwargs):
        dfs = [v.to_df() for v in variables]
        data = pd.concat(dfs, axis=0, sort=True).reset_index(drop=True)
        data = data.rename(columns={name: 'amplitude'})
        return cls(name, data, source=variables[0].source, **kwargs)

    def select_rows(self, rows):
        """Truncate internal arrays to keep only the specified rows.

        Parameters
        ----------
        rows : array_like
            An integer or boolean array identifying the indices
            of rows to keep.
        """
        self.values = self.values.iloc[rows]
        self.index = self.index.iloc[rows, :]
        for prop in self._property_columns:
            vals = getattr(self, prop)[rows]
            setattr(self, prop, vals)


class SparseRunVariable(SimpleVariable):
    """A sparse representation of a single column of events.

    Parameters
    ----------
    name : str
        Name of the column.
    data : :obj:`pandas.DataFrame`
        A pandas DataFrame minimally containing the columns
        'onset', 'duration', and 'amplitude'.
    run_info : list
        A list of RunInfo objects carrying information about
        all runs represented in the Variable.
    source : str
        The type of BIDS variable file the data were extracted
        from. Must be one of: 'events', 'physio', 'stim', 'regressors',
        'scans', 'sessions', 'participants', or 'beh'.
    kwargs : dict
        Optional keyword arguments passed onto superclass.
    """

    _property_columns = {'onset', 'duration'}

    def __init__(self, name, data, run_info, source, **kwargs):
        if hasattr(run_info, 'duration'):
            run_info = [run_info]
        if not isinstance(run_info, list):
            raise TypeError("We expect a list of run_info, got %s"
                            % repr(run_info))
        self.run_info = run_info
        for sc in self._property_columns:
            setattr(self, sc, data.pop(sc).values)
        super(SparseRunVariable, self).__init__(name, data, source, **kwargs)

    def get_duration(self):
        """Return the total duration of the Variable's run(s). """
        return sum([r.duration for r in self.run_info])

    def to_dense(self, sampling_rate=None):
        """Convert the current sparse column to a dense representation.

        If sampling_rate is not provided, the largest interval able to
        faithfully represent all onsets and durations will be determined.
        The sampling rate is the reciprocal of that interval.

        Parameters
        ----------
        sampling_rate : float or None
            Sampling rate (in Hz) to use when constructing the DenseRunVariable

        Returns
        -------
        DenseRunVariable
        """
        # Cast onsets and durations to milliseconds
        onsets = np.round(self.onset * 1000).astype(int)
        durations = np.round(self.duration * 1000).astype(int)
        gcd = np.gcd.reduce(np.r_[onsets, durations])
        bin_sr = 1000. / gcd

        # never use a computed SR smaller than the requested one, because
        # when events are widely-spaced and timing is very regular, this can
        # result in a nasty loss of precision in the resampling step.
        if sampling_rate is not None:
            bin_sr = max(bin_sr, sampling_rate)

        duration = int(math.ceil(bin_sr * self.get_duration()))
        ts = np.zeros(duration, dtype=self.values.dtype)

        onsets = np.round(self.onset * bin_sr).astype(int)
        durations = np.round(self.duration * bin_sr).astype(int)

        run_i, start, last_ind = 0, 0, 0
        for i, val in enumerate(self.values.values):
            if onsets[i] < last_ind:
                start += self.run_info[run_i].duration * bin_sr
                run_i += 1
            _onset = int(start + onsets[i])
            _offset = int(_onset + durations[i])
            if _onset >= duration:
                warnings.warn("The onset time of a variable seems to exceed "
                              "the runs duration, hence runs are incremented "
                              "by one internally.")
            ts[_onset:_offset] = val
            last_ind = onsets[i]

        run_info = list(self.run_info)
        dense_var = DenseRunVariable(
            name=self.name,
            values=ts,
            run_info=run_info,
            source=self.source,
            sampling_rate=bin_sr)

        if sampling_rate is not None and bin_sr != sampling_rate:
            dense_var.resample(sampling_rate, inplace=True)

        return dense_var

    def _extract_entities(self):
        # Get all entities common to all runs. The super method already does
        # this for entities that show up in filenames, so we just add the
        # ones that show up in the RunInfo tuples, as those include metadata.
        ent_items = [run.entities.items() for run in self.run_info]
        entities = reduce(lambda x, y: x & y, ent_items, ent_items[0])
        base_ents = super()._extract_entities()
        return dict(entities, **base_ents)

    @classmethod
    def _merge(cls, variables, name, **kwargs):
        run_info = list(chain(*[v.run_info for v in variables]))
        return super(SparseRunVariable, cls)._merge(variables, name,
                                                    run_info=run_info,
                                                    **kwargs)


class DenseRunVariable(BIDSVariable):
    """A dense representation of a single column.

    Parameters
    ----------
    name : :obj:`str`
        The name of the column.
    values : :obj:`numpy.ndarray`
        The values/amplitudes to store.
    run_info : :obj:`list`
        A list of RunInfo objects carrying information about all runs
        represented in the Variable.
    source : {'events', 'physio', 'stim', 'regressors', 'scans', 'sessions', 'participants', 'beh'}
        The type of BIDS variable file the data were extracted from.
    sampling_rate : :obj:`float`
        Optional sampling rate (in Hz) to use. Must match the sampling rate used
        to generate the values. If None, the collection's sampling rate will be used.
    """

    def __init__(self, name, values, run_info, source, sampling_rate):

        values = pd.DataFrame(values)

        if hasattr(run_info, 'duration'):
            run_info = [run_info]
        self.run_info = run_info
        self.sampling_rate = sampling_rate
        self.index = self._build_entity_index(run_info, sampling_rate)

        super(DenseRunVariable, self).__init__(name, values, source)

    def split(self, grouper):
        """Split the current DenseRunVariable into multiple columns.

        Parameters
        ----------
        grouper : :obj:`pandas.DataFrame`
            Binary DF specifying the design matrix to use for splitting. Number
            of rows must match current ``DenseRunVariable``; a new ``DenseRunVariable``
            will be generated for each column in the grouper.

        Returns
        -------
        A list of DenseRunVariables, one per unique value in the grouper.
        """
        values = grouper.values * self.values.values
        df = pd.DataFrame(values, columns=grouper.columns)
        return [DenseRunVariable(name='%s.%s' % (self.name, name),
                                 values=df[name].values,
                                 run_info=self.run_info,
                                 source=self.source,
                                 sampling_rate=self.sampling_rate)
                for i, name in enumerate(df.columns)]
        
    def _build_entity_index(self, run_info, sampling_rate, match_vol=False):
        """Build the entity index from run information. """

        index = []
        _timestamps = []
        for run in run_info:
            if match_vol:
                # If TR, fix reps to n_vols to ensure match
                reps = run.n_vols
            else:
                reps = int(math.ceil(run.duration * sampling_rate))

            interval = int(round(1000. / sampling_rate))            
            ent_vals = list(run.entities.values())
            df = pd.DataFrame([ent_vals] * reps, columns=list(run.entities.keys()))
            ts = pd.date_range(0, periods=len(df), freq='%sms' % interval)
            _timestamps.append(ts.to_series())
            index.append(df)
        self.timestamps = pd.concat(_timestamps, axis=0, sort=True)
        return pd.concat(index, axis=0, sort=True).reset_index(drop=True)

    def resample(self, sampling_rate, inplace=False, kind='linear'):
        """Resample the Variable to the specified sampling rate.

        Parameters
        ----------
        sampling_rate : :obj:`int`, :obj:`float`
            Target sampling rate (in Hz).
        inplace : :obj:`bool`, optional
            If True, performs resampling in-place. If False, returns a resampled
            copy of the current Variable. Default is False.
        kind : {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'}
            Argument to pass to :obj:`scipy.interpolate.interp1d`; indicates
            the kind of interpolation approach to use. See interp1d docs for
            valid values. Default is 'linear'.
        """
        if not inplace:
            var = self.clone()
            var.resample(sampling_rate, True, kind)
            return var
        
        match_vol = False
        if sampling_rate == 'TR':
            match_vol = True
            sampling_rate = 1. / self.run_info[0].tr
        
        if sampling_rate == self.sampling_rate:
            return
        
        n = len(self.index)

        self.index = self._build_entity_index(self.run_info, sampling_rate, match_vol)
        
        x = np.arange(n)
        num = len(self.index)

        if sampling_rate < self.sampling_rate:
            # Downsampling, so filter the signal
            from scipy.signal import butter, filtfilt
            # cutoff = new Nyqist / old Nyquist
            b, a = butter(5, (sampling_rate / 2.0) / (self.sampling_rate / 2.0),
                          btype='low', output='ba', analog=False)
            y = filtfilt(b, a, self.values.values.ravel())
        else:
            y = self.values.values.ravel()

        from scipy.interpolate import interp1d
        f = interp1d(x, y, kind=kind)
        x_new = np.linspace(0, n - 1, num=num)
        self.values = pd.DataFrame(f(x_new))
        assert len(self.values) == len(self.index)

        self.sampling_rate = sampling_rate

    def to_df(self, condition=True, entities=True, timing=True, sampling_rate=None):
        """Convert to a DataFrame, with columns for name and entities.

        Parameters
        ----------
        condition : :obj:`bool`
            If True, adds a column for condition name, and names the amplitude
            column 'amplitude'. If False, returns just onset, duration, and
            amplitude, and gives the amplitude column the current column name.
        entities : :obj:`bool`
            If True, adds extra columns for all entities.
        timing : :obj:`bool`
            If True, includes onset and duration columns (even though events are
            sampled uniformly). If False, omits them.
        """
        if sampling_rate not in (None, self.sampling_rate):
            return self.resample(sampling_rate).to_df(condition, entities)

        df = super(DenseRunVariable, self).to_df(condition, entities)

        if timing:
            df['onset'] = self.timestamps.values.astype(float) / 1e+9
            df['duration'] = 1. / self.sampling_rate

        return df

    @classmethod
    def _merge(cls, variables, name, sampling_rate=None, **kwargs):

        if not isinstance(sampling_rate, int):
            rates = set([v.sampling_rate for v in variables])
            if len(rates) == 1:
                sampling_rate = list(rates)[0]
            else:
                if sampling_rate == 'auto':
                    sampling_rate = max(rates)
                else:
                    msg = ("Cannot merge DenseRunVariables (%s) with different"
                           " sampling rates (%s). Either specify an integer "
                           "sampling rate to use for all variables, or set "
                           "sampling_rate='highest' to use the highest sampling"
                           " rate found." % (name, rates))
                    raise ValueError(msg)

        variables = [v.resample(sampling_rate) for v in variables]
        values = pd.concat([v.values for v in variables], axis=0, sort=True)
        run_info = list(chain(*[v.run_info for v in variables]))
        source = variables[0].source
        return DenseRunVariable(
            name=name,
            values=values,
            run_info=run_info,
            source=source,
            sampling_rate=sampling_rate)


def merge_variables(variables, **kwargs):
    """Merge/concatenate a list of variables along the row axis.

    Parameters
    ----------
    variables : :obj:`list`
        A list of Variables to merge.
    kwargs
        Optional keyword arguments to pass onto the class-specific merge() call.
        Possible args:
            - sampling_rate (int, str): The sampling rate to use if resampling
              of DenseRunVariables is necessary for harmonization. If
              'highest', the highest sampling rate found will be used. This
              argument is only used when passing DenseRunVariables in the
              variables list.

    Returns
    -------
    A single BIDSVariable of the same class as the input variables.

    Notes
    -----
    - Currently, this function only support homogeneously-typed lists. In
      future, it may be extended to support implicit conversion.
    - Variables in the list must all share the same name (i.e., it is not
      possible to merge two different variables into a single variable.)
    """

    classes = set([v.__class__ for v in variables])
    if len(classes) > 1:
        raise ValueError("Variables of different classes cannot be merged. "
                         "Variables passed are of classes: %s" % classes)

    sources = set([v.source for v in variables])
    if len(sources) > 1:
        raise ValueError("Variables extracted from different types of files "
                         "cannot be merged. Sources found: %s" % sources)

    return list(classes)[0].merge(variables, **kwargs)
