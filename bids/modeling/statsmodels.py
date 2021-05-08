"""BIDS-StatsModels functionality."""

import json
from collections import namedtuple, OrderedDict, Counter, defaultdict
import itertools
from functools import reduce

import numpy as np
import pandas as pd

from bids.layout import BIDSLayout
from bids.utils import matches_entities, convert_JSON, listify
from bids.variables import (BIDSVariableCollection, SparseRunVariable,
                            merge_collections)
from bids.modeling import transformations as tm
from .model_spec import create_model_spec


# Only entities in this list can be used in grouping
VALID_GROUPING_ENTITIES = {'run', 'session', 'subject', 'task', 'contrast'}


def validate_model(model):
    """Validate a BIDS-StatsModel structure.

    Parameters
    ----------
    model : dict
        A full BIDS-StatsModels specification document loaded from JSON.

    Returns
    -------
    True if the model passes validation. Raises an exception otherwise.
    """
    # Identify non-unique names
    names = Counter([n['name'] for n in model['nodes']])
    duplicates = [n for n, count in names.items() if count > 1]
    if duplicates:
        raise ValueError("Non-unique node names found: '{}'. Please ensure"
                            " all nodes in the model have unique names."
                            .format(duplicates))
    if 'edges' in model:
        for edge in model['edges']:
            if edge['src'] not in names:
                raise ValueError("Missing source node: '{}'".format(edge['src']))
            if edge['dst'] not in names:
                raise ValueError("Missing destination node: '{}'".format(edge['dst']))
    return True


BIDSStatsModelsEdge = namedtuple('BIDSStatsModelsEdge', ('src', 'dst', 'groupby'))


ContrastInfo = namedtuple('ContrastInfo', ('name', 'conditions', 'weights',
                                           'type', 'entities'))


class BIDSStatsModelsGraph:
    """Rooted graph structure representing the contents of a BIDS-StatsModel file.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout` or str
        A BIDSLayout instance or path to pass on
        to the BIDSLayout initializer.
    model : str or dict
        A BIDS model specification. Can either be a
        string giving the path of the JSON model spec, or an already-loaded
        dict containing the model info.
    """

    def __init__(self, layout, model):

        if not isinstance(layout, BIDSLayout):
            layout = BIDSLayout(layout)
        self.layout = layout
        self.model = self._load_model(model)
        self.nodes = self._load_nodes(self.model)
        self.edges = self._load_edges(self.model, self.nodes)
        self._root_node = self.model.get('root', list(self.nodes.values())[0])

    def __getitem__(self, key):
        '''Alias for get_node(key).'''
        return self.get_node(key)

    @property
    def root_node(self):
        """Returns the graph's root node."""
        return self._root_node

    @staticmethod
    def _load_model(model, validate=True):
        # Load model info from JSON and do some validation
        if isinstance(model, str):
            with open(model, 'r', encoding='utf-8') as fobj:
                model = json.load(fobj)
        # Convert JSON from CamelCase to snake_case keys
        model = convert_JSON(model)
        if validate:
            validate_model(model)
        return model

    @staticmethod
    def _load_nodes(model):
        nodes = OrderedDict()
        for node_args in model['nodes']:
            node = BIDSStatsModelsNode(**node_args)
            nodes[node.name] = node
        return nodes

    @staticmethod
    def _load_edges(model, nodes):
        """Build edges between nodes."""
        edges = model.get('edges', [])
        # assume we have an optional top-level 'Pipeline' field
        if not edges or model.get('pipeline', False):
            node_vals = list(nodes.values())
            for i in range(1, len(node_vals)):
                # by default, we loop over contrast and the level of the
                # receiving node.
                groupby = ['contrast']
                if node_vals[i].level != 'dataset':
                    groupby.append(node_vals[i].level)
                edges.append({
                    'src': node_vals[i-1].name,
                    'dst': node_vals[i].name,
                    'groupby': groupby
                })

        for edge in edges:
            src_node, dst_node = nodes[edge['src']], nodes[edge['dst']]
            edge = BIDSStatsModelsEdge(src_node, dst_node, edge['groupby'])
            src_node.add_child(edge)
            dst_node.add_parent(edge)

        return edges

    def get_node(self, name):
        """Return the named node.

        Parameters
        ----------
        name : str
            The name of the node to retrieve (as defined in the BIDS-StatsModel
            document).

        Returns
        -------
        A BIDSStatsModelsNode instance.
        """
        if not name in self.nodes:
            raise KeyError('There is no node with the name "{}".'.format(name))
        return self.nodes[name]

    def load_collections(self, nodes=None, drop_na=False, **kwargs):
        """Load collections in all nodes.

        Parameters
        ----------
        nodes : list
            Optional list of nodes to set up. Each element must be a string
            giving the (unique) name of the node, Nodes that do not match
            will be silently skipped.
        drop_na : bool
            Boolean indicating whether or not to automatically
            drop events that have a n/a amplitude when reading in data
            from event files.
        """
        # Use inputs from BIDS-StatsModel document, and update with kwargs
        selectors = self.model.get('input', {}).copy()
        selectors.update(kwargs)

        for node in self.nodes.values():

            # Skip any nodes whose names don't match list
            if nodes is not None and node.name not in nodes:
                continue

            # Currently allowing scan_length to be passed as a hack; remove for
            # levels above run.
            node_kwargs = selectors.copy()
            if node.level != 'run':
                node_kwargs.pop('scan_length', None)

            collections = self.layout.get_collections(node.level, drop_na=drop_na,
                                                      **node_kwargs)
            node.add_collections(collections)


class BIDSStatsModelsNode:
    """Represents a single node in a BIDS-StatsModel graph.

    Parameters
    ----------
    level : str
        The BIDS keyword to use as the grouping variable; must be one of
        ['run', 'session', 'subject', or 'dataset'].
    name : str
        Name to assign to the node.
    transformations : list
        List of BIDS-Model transformations to apply.
    model : dict
        The 'model' part of the BIDS-StatsModels specification.
    contrasts : list
        List of contrasts to apply to the parameter estimates generated when
        the model is fit.
    dummy_contrasts : dict
        Optional dictionary specifying which conditions to create indicator
        contrasts for. Dictionary must include a "type" key ('t' or 'FEMA'),
        and optionally a subset of "conditions".
    groupby: [str]
        Optional list of strings giving the names of entities that define the
        grouping structure for all variables. The current node will be executed
        separately for each unique combination of levels specified in groupby.
        For example, if groupby=['contrast', 'subject'], and there are 2
        contrasts and 3 subjects, then there will be 6 separate iterations, and
        the returned list will have 6 elements. Any value passed here will be
        overridden if one is passed when run() is called on a node.
    """

    def __init__(self, level, name, transformations=None, model=None,
                 contrasts=None, dummy_contrasts=False, groupby=None):
        self.level = level.lower()
        self.name = name
        self.model = model or {}
        self.transformations = transformations or []
        self.contrasts = contrasts or []
        self.dummy_contrasts = dummy_contrasts
        self.groupby = groupby or []
        self._collections = []
        self._group_data = []
        self.children = []
        self.parents = []

    @staticmethod
    def _build_groups(objects, groupby):
        """Group list of objects into bins defined by specified entities.

        Parameters
        ----------
        objects : list
            List of objects containing an .entities dictionary as an attribute.
        groupby : list of str
            List of strings indicating which entities to group on.

        Returns
        -------
        A dictionary, where the keys are tuples of tuples and the values are lists
        containing subsets of the original object list. Each outer tuple of the keys
        represents a particular entity/value combination; each inner tuple has two
        elements, where the first is the name of the entity and the second is the value.

        Notes
        -----
        * The passed objects can be of any type, but alll elements in the list must
        have an implemented .entities dictionary.
        * Objects that do not have defined values for all of the requested
        `entities` will be silently ignored.

        Examples
        --------
        >>> bidsfile = namedtuple('bidsfile', ('entities',))
        >>> bidsfile1 = bidsfile(entities={'subject': '01'})
        >>> bidsfile2 = bidsfile(entities={'subject': '02'})
        >>> groups = {(('subject', '01'),): [bidsfile1],
        ...           (('subject', '02'),): [bidsfile2]}
        >>> BIDSStatsModelsNode._build_groups(
        ...     [bidsfile1, bidsfile2],
        ...     ['subject']) == groups
        True
        """
        if not groupby:
            return {(): objects}

        groups = defaultdict(list)

        # sanitize grouping entities, otherwise weird things can happen
        groupby = list(set(groupby) & VALID_GROUPING_ENTITIES)

        # Get unique values in each grouping variable
        entities = [obj.entities for obj in objects]
        df = pd.DataFrame.from_records(entities).loc[:, groupby]
        unique_vals = {col: df[col].dropna().unique().tolist() for col in groupby}

        # Note: we can't just naively bucket objects based on the values of the
        # grouping entities, because an object may have undefined values for
        # one or more grouping variables. So we "fill in" the missing values by
        # Looping over objects. For each one, identify all grouping variables
        # with missing values. Loop over the cartesian product of the unique
        # values of the missing entities, combining the permutation values with
        # the base (present) grouping entities. Sort on this to produce the
        # group key.
        for i, row in df.iterrows():

            defined = df.columns[df.iloc[i].notnull()].tolist()
            base_ents = [(k, row[k]) for k in defined]
            missing = df.columns[df.iloc[i].isnull()].tolist()

            records = []

            if missing:
                product = itertools.product(*[unique_vals[col] for col in missing])
                for perm in product:
                    fill_ents = [(k, v) for (k, v) in dict(zip(missing, perm)).items()]
                    records.append(base_ents + fill_ents)
            else:
                records.append(base_ents)

            for rec in records:
                grp_key = tuple(sorted(rec, key=lambda x: x[0]))
                groups[grp_key].append(objects[i])

        return groups

    def run(self, inputs=None, groupby=None, force_dense=True,
              sampling_rate='TR', invalid_contrasts='drop', **filters):
        """Execute node with provided inputs.

        Parameters
        ----------
        inputs: [[ContrastInfo]]
            List of list of ContrastInfo tuples passed from the previous node.
            Each element in the outer list maps to the output of a unit at the
            previous level; each element in the inner list is a ContrastInfo
            tuple. E.g., if contrast information is being passed from run-level
            to subject-level, each outer element is a run.
        groupby: [str]
            Optional list of strings giving the names of entities that define
            the grouping structure for all variables. The current node will be
            executed separately for each unique combination of levels specified
            in groupby. For example, if groupby=['contrast', 'subject'], and
            there are 2 contrasts and 3 subjects, then there will be 6 separate
            iterations, and the returned list will have 6 elements. If None is
            passed, the value set at node initialization (if any) will be used.
        force_dense: bool
            If True, the returned design matrices contained in ModelSpec
            instances will represent time in a dense (i.e., uniform temporal
            sampling) format. If False, the returned format will be sparse if
            all available variables are sparse, and dense otherwise. Ignored
            if none of the variables at this node are run-level.
        sampling_rate: str, float
            The sampling rate to use for timeseries resampling when working
            with run-level variables and returning dense output. Ignored if
            there are no run-level variables, or if force_dense is False and
            all available variables are sparse.
        invalid_contrasts: str
            Indicates how to handle invalid contrasts--i.e., ones where the
            specification contains variables that aren't found at run-time.
            Valid values:
                * 'drop' (default): Drop invalid contrasts, retain the rest.
                * 'ignore': Keep invalid contrasts despite the missing variables.
                * 'error': Raise an error.
        filters: dict
            Optional keyword arguments used to constrain the subset of the data
            that's processed. E.g., passing subject='01' will process and
            return data only for files associated with subject '01'.

        Returns
        -------
        A list of BIDSStatsModelsNodeOutput instances.
        """

        inputs = inputs or []
        collections = self._collections
        groupby = groupby or self.groupby
        groupby = listify(groupby)

        # Filter inputs and collections if needed
        if filters:
            inputs = [i for i in inputs if matches_entities(i, filters)]
            collections = [c for c in collections if matches_entities(c, filters)]

        # group all collections and inputs
        all_objects = inputs + collections
        groups = self._build_groups(all_objects, groupby)

        results = []

        for grp_ents, grp_objs in list(groups.items()):

            # split group's objects into inputs and collections
            grp_inputs, grp_colls = [], []
            for obj in grp_objs:
                if isinstance(obj, BIDSVariableCollection):
                    grp_colls.append(obj)
                else:
                    grp_inputs.append(obj)

            node_output = BIDSStatsModelsNodeOutput(
                node=self, entities=dict(grp_ents), collections=grp_colls,
                inputs=grp_inputs, force_dense=force_dense,
                sampling_rate=sampling_rate, invalid_contrasts=invalid_contrasts)
            results.append(node_output)

        return results

    def add_child(self, edge):
        """Add an edge to a child node.

        Parameters
        ----------
        edge : BIDSStatsModelsEdge
            An edge to add to the list of children.
        """
        self.children.append(edge)

    def add_parent(self, edge):
        """Add an edge to a parent node.

        Parameters
        ----------
        edge : BIDSStatsModelsEdge
            An edge to add to the list of children.
        """
        self.parents.append(edge)   

    def add_collections(self, collections):
        """Add BIDSVariableCollections (i.e., predictors) to the current node.

        Parameters
        ----------
        collections : [BIDSVariableCollection]
            List of BIDSVariableCollection objects to add to current node.

        Notes
        -----
        No checking for redundancy is performed, so if load_collections() is
        invoked multiple times with overlapping selectors, redundant predictors
        are likely to be stored internally.
        """
        self._collections.extend(collections)

    def get_collections(self, **filters):
        """Returns BIDSVariableCollections at the current node.
        Parameters
        ----------
        filters : dict
            Optional keyword filters used to constrain which of the available
            collections get returned (e.g., passing subject=['01', '02'] will
            return collections for only subjects '01' and '02').
        Returns
        -------
        list of BIDSVariableCollection instances
            One instance per unit of the current analysis level (e.g., if
            level='run', each element in the list represents the collection
            for a single run).
        """
        # Keeps only collections that match target entities, and also removes
        # those keys from the kwargs dict.
        return [c for c in self._collections if matches_entities(c, filters)]


class BIDSStatsModelsNodeOutput:
    """Represents a single node in a BIDSStatsModelsGraph.

    Parameters
    ----------
    node : BIDSStatsModelsNode
        The node that generated the current instance.
    entities : dict
        Dictionary of entities/metadata applicable to the current node output.
    collections : [BIDSVariableCollection]
        List of BIDSVariableCollection instances needed too generate outputs.
    inputs : [ContrastInfo]
        List of ContrastInfo instances used to generate outputs.
    force_dense: bool
        If True, the returned design matrices contained in ModelSpec instances
        will represent time in a dense (i.e., uniform temporal sampling) format.
        If False, the returned format will be sparse if all available variables
        are sparse, and dense otherwise. Ignored if none of the variables at
        this node are run-level.
    sampling_rate: str, float
        The sampling rate to use for timeseries resampling when working with
        run-level variables and returning dense output. Ignored if there are no
        run-level variables, or if force_dense is False and all available
        variables are sparse.
    invalid_contrasts: str
        Indicates how to handle invalid contrasts--i.e., ones where the
        specification contains variables that aren't found at run-time.
        Valid values:
            * 'drop' (default): Drop invalid contrasts, retain the rest.
            * 'ignore': Keep invalid contrasts despite the missing variables.
            * 'error': Raise an error.
    """
    def __init__(self, node, entities={}, collections=None, inputs=None,
                 force_dense=True, sampling_rate='TR', invalid_contrasts='drop'):

        collections = collections or []
        inputs = inputs or []

        self.node = node
        self.entities = entities
        self.force_dense = force_dense
        self.sampling_rate = sampling_rate
        self.invalid_contrasts = invalid_contrasts

        dfs = self._collections_to_dfs(collections)

        if inputs:
            dfs.append(self._inputs_to_df(inputs))

        # merge all the DataFrames into one DF to rule them all
        def merge_dfs(a, b):
            on = list(set(a.columns) & set(b.columns) & VALID_GROUPING_ENTITIES)
            return a.merge(b, on=on)
        df = reduce(merge_dfs, dfs)

        var_names = self.node.model['x']
        # need to settle on some spec-level decision about which case to use
        # here, whether names are insensitive, etc.
        if ('intercept' in var_names and 'intercept' not in df.columns) or \
           ('Intercept' in var_names and 'Intercept' not in df.columns):
            df.insert(0, 'Intercept', 1)

        # separate the design columns from the entity columns
        self.data = df.loc[:, var_names]
        self.metadata = df.loc[:, df.columns.difference(var_names)]
        self.model_spec = create_model_spec(self.data, node.model, self.metadata)

        self.contrasts = self._build_contrasts()

    def _collections_to_dfs(self, collections):
        """Merges collections and converts them to a pandas DataFrame."""
        if not collections:
            return []

        # group all collections by level
        coll_levels = defaultdict(list)
        [coll_levels[coll.level].append(coll) for coll in collections]

        # ugly hack: we need to remove intercepts from the variable list (which
        # is used to filter the collection later), as they can be specified in
        # 'X' but don't exist yet. this business with intercepts really needs
        # a spec-level resolution, at which point we can be more principled.
        var_names = list(set(self.node.model['x']) - {'Intercept', 'intercept'})

        grp_dfs = []
        # merge all collections at each level and export to a DataFrame 
        for level, colls in coll_levels.items():

            # Note: we currently merge _before_ selecting variables. Selecting
            # variables first could be done by passing `variables=all_vars` as
            # an argument on the next line), but we can't do this right now
            # because we can't guarantee that all the variables named in `X`
            # in the model section already exist; some might be created by the
            # transformations.
            coll = merge_collections(colls)

            # apply transformations
            transformations = self.node.transformations
            if transformations:
                tm.TransformerManager().transform(coll, transformations)

            # retain only variables listed in 'X', and skip level if none are left.
            tm.Select(coll, var_names)
            if not coll.variables:
                continue

            # run collections need to be handled separately because to_df()
            # takes extra arguments related to handling of time
            if level == 'run':
                if self.force_dense:
                    coll = coll.to_dense(sampling_rate=self.sampling_rate)
                coll = coll.to_df(sampling_rate=self.sampling_rate)
            else:
                coll = coll.to_df()
            grp_dfs.append(coll)

        return grp_dfs

    def _inputs_to_df(self, inputs):
        # Convert the inputs to a DataFrame and add to list. Each row is
        # an input; each column is either an entity or a contrast name from
        # the previous level.
        if inputs:
            rows = [{**con.entities, 'contrast': con.name} for con in inputs]
            input_df = pd.DataFrame.from_records(rows)
            for i, con in enumerate(inputs):
                if con.name not in input_df.columns:
                    input_df.loc[:, con.name] = 0
                input_df.loc[input_df.index[i], con.name] = 1
        return input_df

    def _build_contrasts(self):
        """Contrast list of ContrastInfo objects based on current state."""
        contrasts = {}
        col_names = set(self.X.columns)
        for con in self.node.contrasts:
            missing_vars = set(con['condition_list']) - col_names
            if missing_vars:
                if self.invalid_contrasts == 'error':
                    msg = ("Variable(s) '{}' specified in condition list for "
                           "contrast '{}', but not found in available data."
                           .format(missing_vars, con['name']))
                    raise ValueError(msg)
                elif self.invalid_contrasts == 'drop':
                    continue
            weights = np.atleast_2d(con['weights'])
            matrix = pd.DataFrame(weights, columns=con['condition_list'])
            test_type = con.get('type', ('t' if len(weights) == 1 else 'F'))
            # Add contrast name to entities; can be used in grouping downstream
            entities = {**self.entities, 'contrast': con['name']}
            ci = ContrastInfo(con['name'], con['condition_list'],
                              con['weights'], test_type, entities)
            contrasts[con['name']] = ci
        
        dummies = self.node.dummy_contrasts
        if dummies:
            conditions = col_names
            if 'conditions' in dummies:
                conditions &= set(dummies['conditions'])
            conditions -= set([c.name for c in contrasts])

            for col_name in conditions:
                if col_name in contrasts:
                    continue
                entities = {**self.entities, 'contrast': col_name}
                ci = ContrastInfo(col_name, [col_name], [1], dummies['type'],
                                  entities)
                contrasts[col_name] = ci

        return list(contrasts.values())

    @property
    def X(self):
        """Return design matrix via the current ModelSpec."""
        return self.model_spec.X
