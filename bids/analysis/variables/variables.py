import numpy as np
import pandas as pd
import math
from copy import deepcopy
from abc import abstractproperty, abstractmethod, abstractclassmethod, ABCMeta
from scipy.interpolate import interp1d
from bids.utils import listify


class BIDSVariable(object):

    ''' Base representation of a column in a BIDS project. '''

    __metaclass__ = ABCMeta

    def __init__(self, name, values):
        self.name = name
        self.values = values

    def clone(self, data=None, **kwargs):
        ''' Clone (deep copy) the current column, optionally replacing its
        data and/or any other attributes.

        Args:
            data (DataFrame, ndarray): Optional new data to substitute into
                the cloned column. Must have same dimensionality as the
                original.
            kwargs (dict): Optional keyword arguments containing new attribute
                values to set in the copy. E.g., passing `name='my_name'`
                would set the `.name` attribute on the cloned instance to the
                passed value.
        '''
        result = deepcopy(self)
        if data is not None:
            if data.shape != self.values.shape:
                raise ValueError("Replacement data has shape %s; must have "
                                 "same shape as existing data %s." %
                                 (data.shape, self.values.shape))
            result.values = pd.Series(data)

        if kwargs:
            for k, v in kwargs.items():
                setattr(result, k, v)

        # Need to update name on Series as well
        result.values.name = kwargs.get('name', self.name)
        return result

    @abstractmethod
    def aggregate(self, unit, level, func):
        pass

    @abstractclassmethod
    def merge(cls, columns, name=None):

        col_names = set([c.name for c in columns])
        if len(col_names) > 1:
            raise ValueError("Columns with different names cannot be merged. "
                             "Column names provided: %s" % col_names)

        if name is None:
            name = columns[0].name

        return cls._merge(columns, name)

    @abstractproperty
    def index(self):
        pass

    def get_grouper(self, groupby='unique_run_id'):
        ''' Return a pandas Grouper object suitable for use in groupby calls.
        Args:
            groupby (str, list): Name(s) of column(s) defining the grouper
                object. Anything that would be valid inside a .groupby() call
                on a pandas structure.
        Returns:
            A pandas Grouper object constructed from the specified columns
                of the current index.
        '''
        return pd.core.groupby._get_grouper(self.index, groupby)[0]

    def apply(self, func, groupby='unique_run_id', *args, **kwargs):
        ''' Applies the passed function to the groups defined by the groupby
        argument. Works identically to the standard pandas df.groupby() call.
        Args:
            func (callable): The function to apply to each group.
            groupby (str, list): Name(s) of column(s) defining the grouping.
            args, kwargs: Optional positional and keyword arguments to pass
                onto the function call.
        '''
        grouper = self.get_grouper(groupby)
        return self.values.groupby(grouper).apply(func, *args, **kwargs)


class SimpleVariable(BIDSVariable):
    ''' Represents a simple design matrix column that has no timing
    information.

    Args:
        name (str): Name of the column.
        data (DataFrame): A pandas DataFrame minimally containing a column
            named 'amplitude' as well as any identifying entities.
    '''

    # Columns that define special properties (e.g., onset, duration). These
    # will be stored separately from the main data object, and are accessible
    # as properties on the SimpleVariable instance.
    _property_columns = set()
    _entity_columns = {'condition', 'amplitude', 'factor'}

    def __init__(self, name, data):

        for sc in self._property_columns:
            setattr(self, sc, data[sc].values)

        ent_cols = list(set(data.columns) - self._entity_columns -
                        self._property_columns)
        self.entities = data.loc[:, ent_cols]

        values = data['amplitude'].reset_index(drop=True)
        values.name = name

        super(SimpleVariable, self).__init__(name, values)

    def aggregate(self, unit, func='mean'):

        levels = ['run', 'session', 'subject']
        groupby = set(levels[levels.index(unit):]) & set(self.entities.columns)
        groupby = list(groupby)

        entities = self.entities.loc[:, groupby].reset_index(drop=True)
        values = pd.DataFrame({'amplitude': self.values.values})
        data = pd.concat([values, entities], axis=1)
        data = data.groupby(groupby, as_index=False).agg(func)
        return SimpleVariable(self.name, data)

    def to_df(self, condition=True, entities=True):
        ''' Convert to a DataFrame, with columns for name and entities.
        Args:
            condition (bool): If True, adds a column for condition name, and
                names the amplitude column 'amplitude'. If False, returns just
                onset, duration, and amplitude, and gives the amplitude column
                the current column name.
            entities (bool): If True, adds extra columns for all entities.
        '''
        amp = 'amplitude' if condition else self.name
        data = pd.DataFrame({amp: self.values.values.ravel()})

        for sc in self._property_columns:
            data[sc] = getattr(self, sc)

        if condition:
            data['condition'] = self.name

        if entities:
            ent_data = self.entities.reset_index(drop=True)
            data = pd.concat([data, ent_data], axis=1)

        return data

    def split(self, grouper):
        ''' Split the current SparseRunVariable into multiple columns.
        Args:
            grouper (iterable): list to groupby, where each unique value will
                be taken as the name of the resulting column.
        Returns:
            A list of SparseRunVariables, one per unique value in the
            grouper.
        '''
        data = self.to_df(condition=True, entities=False)
        data = data.drop('condition', axis=1)
        # data = pd.DataFrame(dict(onset=self.onset, duration=self.duration,
        #                          amplitude=self.values.values))
        # data = pd.concat([data, self.index.reset_index(drop=True)], axis=1)

        subsets = []
        for i, (name, g) in enumerate(data.groupby(grouper)):
            name = '%s.%s' % (self.name, name)
            col = self.__class__(name, g)
            subsets.append(col)
        return subsets

    @property
    def index(self):
        ''' An index of all named entities. '''
        return self.entities

    @classmethod
    def _merge(cls, variables, name):
        dfs = [v.to_df() for v in variables]
        data = pd.concat(dfs, axis=0).reset_index(drop=True)
        data = data.rename(columns={name: 'amplitude'})
        return cls(name, data)


class SparseRunVariable(SimpleVariable):
    ''' A sparse representation of a single column of events.

    Args:
        name (str): Name of the column.
        data (DataFrame): A pandas DataFrame minimally containing the columns
            'onset', 'duration', and 'amplitude'.
        durations (float, list): ???
    '''

    _property_columns = {'onset', 'duration'}

    def __init__(self, name, data, run_info):
        self.run_info = listify(run_info)
        super(SparseRunVariable, self).__init__(name, data)

    def to_dense(self, sampling_rate):
        ''' Convert the current sparse column to a dense representation.
        Returns: A DenseRunVariable. '''
        duration = math.ceil(sampling_rate * self.duration)
        ts = np.zeros(duration)

        onsets = round(self.onset * sampling_rate).astype(int)
        durations = round(self.duration * sampling_rate).astype(int)

        for i, row in enumerate(self.values.values):
            ev_end = onsets[i] + durations[i]
            ts[onsets[i]:ev_end] = row

        return DenseRunVariable(self.name, pd.DataFrame(ts))


class DenseRunVariable(BIDSVariable):
    ''' A dense representation of a single column.

    Args:
        name (str): The name of the column
        values (NDArray): The values/amplitudes to store
        sampling_rate (float): Optional sampling rate (in Hz) to use. Must
            match the sampling rate used to generate the values. If None,
            the collection's sampling rate will be used.
    '''

    def __init__(self, name, values, run_info, sampling_rate):

        values = pd.Series(values, name=name)
        super(DenseRunVariable, self).__init__(name, values)

        if hasattr(run_info, 'duration'):
            run_info = [run_info]
        self.run_info = run_info
        self.sampling_rate = sampling_rate
        self.entities = self.build_entity_index(run_info, sampling_rate)

    @property
    def index(self):
        ''' An index of all named entities. '''
        return self.entities

    def split(self, grouper):
        ''' Split the current DenseRunVariable into multiple columns.
        Args:
            grouper (DataFrame): binary DF specifying the design matrix to
                use for splitting. Number of rows must match current
                DenseRunVariable; a new DenseRunVariable will be generated
                for each column in the grouper.
        Returns:
            A list of DenseRunVariables, one per unique value in the grouper.
        '''
        df = grouper * self.values
        names = df.columns
        return [DenseRunVariable('%s.%s' % (self.name, name),
                                   df[name].values)
                for i, name in enumerate(names)]

    def aggregate(self, unit, func='mean'):

        levels = ['run', 'session', 'subject']
        groupby = set(levels[levels.index(unit):]) & \
            set(self.index.columns)
        groupby = list(groupby)

        entities = self._index.loc[:, groupby].reset_index(drop=True)
        values = pd.DataFrame({'amplitude': self.values.values.ravel()})
        data = pd.concat([values, entities], axis=1)
        data = data.groupby(groupby, as_index=False).agg(func)
        return SimpleVariable(self.name, data)

    @staticmethod
    def build_entity_index(run_info, sampling_rate):
        index = []
        sr = int(round(1000. / sampling_rate))
        for run in run_info:
            reps = int(math.ceil(run.duration * sampling_rate))
            ent_vals = list(run.entities.values())
            data = np.broadcast_to(ent_vals, (reps, len(ent_vals)))
            df = pd.DataFrame(data, columns=list(run.entities.keys()))
            df['time'] = pd.date_range(0, periods=len(df), freq='%sms' % sr)
            index.append(df)
        return pd.concat(index, axis=0).reset_index(drop=True)

    def resample(self, sampling_rate, inplace=False, kind='linear'):
        ''' Resample the Variable to the specified sampling rate.
        Args:
            sampling_rate (int, float): Target sampling rate (in Hz)
            inplace (bool): If True, performs resampling in-place. If False,
                returns a resampled copy of the current Variable.
            kind (str): Argument to pass to scipy's interp1d; indicates the
                kind of interpolation approach to use. See interp1d docs for
                valid values.
        '''
        if not inplace:
            var = self.clone()
            var.resample(sampling_rate, True, kind)
            return var

        if sampling_rate == self.sampling_rate:
            return

        old_sr = self.sampling_rate
        n = len(self.index)

        self.entities = self.build_entity_index(self.run_info, sampling_rate)

        x = np.arange(n)
        num = int(np.ceil(n * sampling_rate / old_sr))

        f = interp1d(x, self.values.values.ravel(), kind=kind)
        x_new = np.linspace(0, n - 1, num=num)
        self.values = pd.DataFrame(f(x_new))

        self.sampling_rate = sampling_rate

    @classmethod
    def _merge(cls, variables, name, sampling_rate=None):

        # If no sampling rate was provided, default to the highest rate found
        if sampling_rate is None:
            rates = [v.sampling_rate for v in variables]
            sampling_rate = max(set(rates))

        variables = [v.clone().resample(sampling_rate) for v in variables]

        dfs = [v.to_df() for v in variables]
        data = pd.concat(dfs, axis=0).reset_index(drop=True)
        data = data.rename(columns={name: 'amplitude'})
        return cls(name, data)

    @classmethod
    def harmonize(cls, variables, sampling_rate=None):
        ''' Harmonize two or more DenseRunVariables so that they share the
        same sampling rate. '''
        pass


def merge_variables(variables):
    classes = set([v.__class__ for v in variables])
    if len(classes) > 1:
        raise ValueError("Columns of different classes cannot be merged. "
                         "Columns passed are of classes: %s" % classes)
    return list(classes)[0].merge(variables)