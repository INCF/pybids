""" Model classes used in BIDSLayouts. """

import re
import os
from pathlib import Path
import warnings
import json
from copy import deepcopy
from itertools import chain
from functools import lru_cache

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy import (Column, String, Boolean, ForeignKey, Table)
from sqlalchemy.orm import reconstructor, relationship, backref, object_session

from ..utils import listify
from .writing import build_path, write_to_file
from ..config import get_option
from .utils import BIDSMetadata

Base = declarative_base()


class LayoutInfo(Base):
    """ Contains information about a BIDSLayout's initialization parameters."""

    __tablename__ = 'layout_info'

    root = Column(String, primary_key=True)
    absolute_paths = Column(Boolean)
    _derivatives = Column(String)
    _config = Column(String)

    def __init__(self, **kwargs):
        init_args = self._sanitize_init_args(kwargs)
        raw_cols = ['root', 'absolute_paths']
        json_cols = ['derivatives', 'config']
        all_cols = raw_cols + json_cols
        missing_cols = set(all_cols) - set(init_args.keys())
        if missing_cols:
            raise ValueError("Missing mandatory initialization args: {}"
                             .format(missing_cols))
        for col in all_cols:
            setattr(self, col, init_args[col])
            if col in json_cols:
                json_data = json.dumps(init_args[col])
                setattr(self, '_' + col, json_data)

    @reconstructor
    def _init_on_load(self):
        for col in ['derivatives', 'config']:
            db_val = getattr(self, '_' + col)
            setattr(self, col, json.loads(db_val))

    def _sanitize_init_args(self, kwargs):
        """ Prepare initialization arguments for serialization """
        if 'root' in kwargs:
            kwargs['root'] = str(Path(kwargs['root']).absolute())

        # Get abspaths
        if kwargs.get('derivatives') not in (None, True, False):
            kwargs['derivatives'] = [
                str(Path(der).absolute())
                for der in listify(kwargs['derivatives'])
                ]

        return kwargs


class Config(Base):
    """Container for BIDS configuration information.

    Parameters
    ----------
    name : str
        The name to give the Config (e.g., 'bids').
    entities : list
        A list of dictionaries containing entity configuration
        information.
    default_path_patterns : list
        Optional list of patterns used to build new paths.
    session : :obj:`sqlalchemy.orm.session.Session` or None
        An optional SQLAlchemy session. If passed,
        the session is used to update the database with any newly created
        Entity objects. If None, no database update occurs.
    """
    __tablename__ = 'configs'

    name = Column(String, primary_key=True)
    _default_path_patterns = Column(String)
    entities = relationship(
        "Entity", secondary="config_to_entity_map",
        collection_class=attribute_mapped_collection('name'))

    def __init__(self, name, entities=None, default_path_patterns=None,
                 session=None):

        self.name = name
        self.default_path_patterns = default_path_patterns
        self._default_path_patterns = json.dumps(default_path_patterns)

        if entities:
            for ent in entities:
                if session is not None:
                    existing = (session.query(Config)
                                .filter_by(name=ent['name']).first())
                else:
                    existing = None
                ent = existing or Entity(**ent)
                self.entities[ent.name] = ent
                if session is not None:
                    session.add_all(list(self.entities.values()))
                    session.commit()

    @reconstructor
    def _init_on_load(self):
        self.default_path_patterns = json.loads(self._default_path_patterns)

    @classmethod
    def load(self, config, session=None):
        """Load a Config instance from the passed configuration data.

        Parameters
        ----------
        config : str or dict
            A string or dict containing configuration information.
            Must be one of:
            * A string giving the name of a predefined config file
                (e.g., 'bids' or 'derivatives')
            * A path to a JSON file containing config information
            * A dictionary containing config information
        session : :obj:`sqlalchemy.orm.session.Session` or None
            An optional SQLAlchemy Session instance.
            If passed, the session is used to check the database for (and
            return) an existing Config with name defined in config['name'].

        Returns
        -------
        A Config instance.
        """

        # XXX 0.14: Disable extension_initial_dot branching
        extension_initial_dot = get_option('extension_initial_dot')
        if config == "bids" and not extension_initial_dot:
            if extension_initial_dot is None:
                warnings.warn("The 'extension' entity currently excludes the leading dot ('.'). "
                              "As of version 0.14.0, it will include the leading dot. To suppress "
                              "this warning and include the leading dot, use "
                              "`bids.config.set_option('extension_initial_dot', True)`.",
                              FutureWarning)
            config = "bids-nodot"

        if isinstance(config, str):
            config_paths = get_option('config_paths')
            if config in config_paths:
                config = config_paths[config]
            if not os.path.exists(config):
                raise ValueError("{} is not a valid path.".format(config))
            else:
                with open(config, 'r') as f:
                    config = json.load(f)

        # Return existing Config record if one exists
        if session is not None:
            result = session.query(Config).filter_by(name=config['name']).first()
            if result:
                # XXX 0.14: Remove check
                if config["name"] == "bids":
                    old_pattern = result.entities["extension"].pattern
                    new_pattern = next(ent['pattern'] for ent in config['entities']
                                       if ent['name'] == 'extension')
                    if new_pattern != old_pattern:
                        warnings.warn(
                            "Cannot modify extension_initial_dot option after initialization. "
                            "Set option immediately after ``import bids``.")
                return result

        return Config(session=session, **config)


class BIDSFile(Base):
    """Represents a single file or directory in a BIDS dataset.

    Parameters
    ----------
    filename : str
        The path to the corresponding file.
    """
    __tablename__ = 'files'

    path = Column(String, primary_key=True)
    filename = Column(String)
    dirname = Column(String)
    entities = association_proxy("tags", "value")
    is_dir = Column(Boolean)
    class_ = Column(String(20))

    _associations = relationship('BIDSFile', secondary='associations',
                                 primaryjoin='FileAssociation.dst == BIDSFile.path',
                                 secondaryjoin='FileAssociation.src == BIDSFile.path')

    __mapper_args__ = {
        'polymorphic_on': class_,
        'polymorphic_identity': 'file'
    }

    def __init__(self, filename):
        self.path = filename
        self.filename = os.path.basename(self.path)
        self.dirname = os.path.dirname(self.path)
        self.is_dir = not self.filename

    def __getattr__(self, attr):
        # Ensures backwards compatibility with old File_ namedtuple, which is
        # deprecated as of 0.7.
        # _ check first to not mask away access to __setstate__ etc.
        # AFAIK None of the entities are allowed to start with _ anyways
        # so the check is more generic than __
        if not attr.startswith('_') and attr in self.entities:
            warnings.warn("Accessing entities as attributes is deprecated as "
                          "of 0.7. Please use the .entities dictionary instead"
                          " (i.e., .entities['%s'] instead of .%s."
                          % (attr, attr))
            return self.entities[attr]
        raise AttributeError("%s object has no attribute named %r" %
                             (self.__class__.__name__, attr))

    def __repr__(self):
        return "<{} filename='{}'>".format(self.__class__.__name__, self.path)

    def __fspath__(self):
        return self.path

    @property
    @lru_cache()
    def relpath(self):
        """Return path relative to layout root"""
        root = object_session(self).query(LayoutInfo).first().root
        return str(Path(self.path).relative_to(root))

    def get_associations(self, kind=None, include_parents=False):
        """Get associated files, optionally limiting by association kind.

        Parameters
        ----------
        kind : str
            The kind of association to return (e.g., "Child").
            By default, all associations are returned.
        include_parents : bool
            If True, files related through inheritance
            are included in the returned list. If False, only directly
            associated files are returned. For example, a file's JSON
            sidecar will always be returned, but other JSON files from
            which the sidecar inherits will only be returned if
            include_parents=True.

        Returns
        -------
        list
            A list of BIDSFile instances.
        """
        if kind is None and not include_parents:
            return self._associations

        session = object_session(self)
        q = (session.query(BIDSFile)
             .join(FileAssociation, BIDSFile.path == FileAssociation.dst)
             .filter_by(src=self.path))

        if kind is not None:
            q = q.filter_by(kind=kind)

        associations = q.all()

        if not include_parents:
            return associations

        def collect_associations(results, bidsfile):
            results.append(bidsfile)
            for p in bidsfile.get_associations('Child'):
                results = collect_associations(results, p)
            return results

        return list(chain(*[collect_associations([], bf) for bf in associations]))

    def get_metadata(self):
        """Return all metadata associated with the current file. """
        md = BIDSMetadata(self.path)
        md.update(self.get_entities(metadata=True))
        return md

    def get_entities(self, metadata=False, values='tags'):
        """Return entity information for the current file.

        Parameters
        ----------
        metadata : bool or None
            If False (default), only entities defined
            for filenames (and not those found in the JSON sidecar) are
            returned. If True, only entities found in metadata files (and not
            defined for filenames) are returned. If None, all available
            entities are returned.
        values : str
            The kind of object to return in the dict's values.
            Must be one of:
                * 'tags': Returns only the tagged value--e.g., if the key
                is "subject", the value might be "01".
                * 'objects': Returns the corresponding Entity instance.

        Returns
        -------
        dict
            A dict, where keys are entity names and values are Entity
            instances.
        """
        session = object_session(self)
        query = (session.query(Tag)
                 .filter_by(file_path=self.path)
                 .join(Entity))
        if metadata not in (None, 'all'):
            query = query.filter(Entity.is_metadata == metadata)

        results = query.all()
        if values.startswith('obj'):
            return {t.entity_name: t.entity for t in results}
        return {t.entity_name: t.value for t in results}

    def copy(self, path_patterns, symbolic_link=False, root=None,
             conflicts='fail'):
        """Copy the contents of a file to a new location.

        Parameters
        ----------
        path_patterns : list
            List of patterns used to construct the new
            filename. See :obj:`build_path` documentation for details.
        symbolic_link : bool
            If True, use a symbolic link to point to the
            existing file. If False, creates a new file.
        root : str
            Optional path to prepend to the constructed filename.
        conflicts : str
            Defines the desired action when the output path already exists.
            Must be one of:
                'fail': raises an exception
                'skip' does nothing
                'overwrite': overwrites the existing file
                'append': adds  a suffix to each file copy, starting with 1
        """
        new_filename = build_path(self.entities, path_patterns)
        if not new_filename:
            return None

        if new_filename[-1] == os.sep:
            new_filename += self.filename

        if os.path.isabs(self.path) or root is None:
            path = self.path
        else:
            path = os.path.join(root, self.path)

        if not os.path.exists(path):
            raise ValueError("Target filename to copy/symlink (%s) doesn't "
                             "exist." % path)

        kwargs = dict(path=new_filename, root=root, conflicts=conflicts)
        if symbolic_link:
            kwargs['link_to'] = path
        else:
            kwargs['copy_from'] = path

        write_to_file(**kwargs)


class BIDSDataFile(BIDSFile):
    """Represents a single data file in a BIDS dataset.

    Derived from `BIDSFile` and provides additional functionality such as
    obtaining pandas DataFrame data representation (via `get_df`).
    """

    __mapper_args__ = {
        'polymorphic_identity': 'data_file'
    }

    def get_df(self, include_timing=True, adjust_onset=False,
               enforce_dtypes=True, **pd_args):
        """Return the contents of a tsv file as a pandas DataFrame.

        Parameters
        ----------
        include_timing : bool
            If True, adds an "onset" column to dense
            timeseries files (e.g., *_physio.tsv.gz).
        adjust_onset : bool
            If True, the onset of each sample in a dense
            timeseries file is shifted to reflect the "StartTime" value in
            the JSON sidecar. If False, the first sample starts at 0 secs.
            Ignored if include_timing=False.
        enforce_dtypes : bool
            If True, enforces the data types defined in
            the BIDS spec on core columns (e.g., subject_id and session_id
            must be represented as strings).
        pd_args : dict
            Optional keyword arguments to pass onto pd.read_csv().

        Returns
        -------
        :obj:`pandas.DataFrame`
            A pandas DataFrame.
        """
        import pandas as pd
        import numpy as np

        if enforce_dtypes:
            dtype = {
                'subject_id': str,
                'session_id': str,
                'participant_id': str
            }
        else:
            dtype = None

        # TODO: memoize this for efficiency. (Note: caching is insufficient,
        # because the dtype enforcement will break if we ignore the value of
        # enforce_dtypes).
        suffix = self.entities['suffix']
        header = None if suffix in {'physio', 'stim'} else 'infer'
        self.data = pd.read_csv(self.path, sep='\t', na_values='n/a',
                                dtype=dtype, header=header, **pd_args)

        data = self.data.copy()

        # XXX 0.14: ".tsv.gz" only
        if self.entities['extension'] in ('tsv.gz', '.tsv.gz'):
            md = self.get_metadata()
            # We could potentially include some validation here, but that seems
            # like a job for the BIDS Validator.
            data.columns = md['Columns']
            if include_timing:
                onsets = np.arange(len(data)) / md['SamplingFrequency']
                if adjust_onset:
                    onsets += md['StartTime']
                data.insert(0, 'onset', onsets)

        return data


class BIDSImageFile(BIDSFile):
    """Represents a single neuroimaging data file in a BIDS dataset.

    Derived from `BIDSFile` and provides additional functionality such as
    obtaining nibabel's image file representation (via `get_image`).
    """

    __mapper_args__ = {
        'polymorphic_identity': 'image_file'
    }

    def get_image(self, **kwargs):
        """Return the associated image file (if it exists) as a NiBabel object

        Any keyword arguments are passed to ``nibabel.load``.
        """
        try:
            import nibabel as nb
            return nb.load(self.path, **kwargs)
        except Exception as e:
            raise ValueError("'{}' does not appear to be an image format "
                             "NiBabel can read.".format(self.path)) from e


class BIDSJSONFile(BIDSFile):
    """Represents a single JSON metadata file in a BIDS dataset.

    Derived from `BIDSFile` and provides additional functionality for reading
    the contents of JSON files as either dicts or strings.
    """
    __mapper_args__ = {
        'polymorphic_identity': 'json_file'
    }

    def get_dict(self):
        """Return the contents of the current file as a dictionary. """
        d = json.loads(self.get_json())
        if not isinstance(d, dict):
            raise ValueError("File %s is a json containing %s, not a dict which was expected" % (self.path, type(d)))
        return d

    def get_json(self):
        """Return the contents of the current file as a JSON string. """
        with open(self.path, 'r') as f:
            return f.read()


class Entity(Base):
    """
    Represents a single entity defined in the JSON config.

    Parameters
    ----------
    name : str
        The name of the entity (e.g., 'subject', 'run', etc.)
    pattern : str
        A regex pattern used to match against file names.
        Must define at least one group, and only the first group is
        kept as the match.
    mandatory : bool
        If True, every File _must_ match this entity.
    directory : str
        Optional pattern defining a directory associated
        with the entity.
    dtype : str
        The optional data type of the Entity values. Must be
        one of 'int', 'float', 'bool', or 'str'. If None, no type
        enforcement will be attempted, which means the dtype of the
        value may be unpredictable.
    is_metadata : bool
        Indicates whether or not the Entity is derived
        from JSON sidecars (True) or is a predefined Entity from a
        config (False).
    """
    __tablename__ = 'entities'

    name = Column(String, primary_key=True)
    mandatory = Column(Boolean, default=False)
    pattern = Column(String)
    directory = Column(String, nullable=True)
    is_metadata = Column(Boolean, default=False)
    _dtype = Column(String, default='str')
    files = association_proxy("tags", "value")

    def __init__(self, name, pattern=None, mandatory=False, directory=None,
                 dtype='str', is_metadata=False):
        self.name = name
        self.pattern = pattern
        self.mandatory = mandatory
        self.directory = directory
        self.is_metadata = is_metadata

        if not isinstance(dtype, str):
            dtype = dtype.__name__
        self._dtype = dtype

        self._init_on_load()

    @reconstructor
    def _init_on_load(self):
        if self._dtype not in ('str', 'float', 'int', 'bool'):
            raise ValueError("Invalid dtype '{}'. Must be one of 'int', "
                             "'float', 'bool', or 'str'.".format(self._dtype))
        self.dtype = eval(self._dtype)
        self.regex = re.compile(self.pattern) if self.pattern is not None else None

    def __iter__(self):
        for i in self.unique():
            yield i

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)

        # Directly copy the SQLAlchemy connection before any setattr calls,
        # otherwise failures occur sporadically on Python 3.5 when the
        # _sa_instance_state attribute (randomly!) disappears.
        result._sa_instance_state = self._sa_instance_state

        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k == '_sa_instance_state':
                continue
            new_val = getattr(self, k) if k == 'regex' else deepcopy(v, memo)
            setattr(result, k, new_val)
        return result

    def match_file(self, f):
        """
        Determine whether the passed file matches the Entity.

        Parameters
        ----------
        f : BIDSFile
            The BIDSFile instance to match against.

        Returns
        -------
        the matched value if a match was found, otherwise None.
        """
        if self.regex is None:
            return None
        m = self.regex.search(f.path)
        val = m.group(1) if m is not None else None

        return self._astype(val)

    def unique(self):
        """Return all unique values/levels for the current entity.
        """
        return list(set(self.files.values()))

    def count(self, files=False):
        """Return a count of unique values or files.

        Parameters
        ----------
        files : bool
            When True, counts all files mapped to the Entity.
            When False, counts all unique values.

        Returns
        -------
        int
            Count of unique values or files.
        """
        return len(self.files) if files else len(self.unique())

    def _astype(self, val):
        if val is not None and self.dtype is not None:
            val = self.dtype(val)
        return val


class Tag(Base):
    """Represents an association between a File and an Entity.

    Parameters
    ----------
    file : BIDSFile
        The associated BIDSFile.
    entity : Entity
        The associated Entity.
    value : json-serializable type
        The value to store for this file/entity pair. Must be of type
        str, int, float, bool, or any json-serializable structure.
    dtype : str
        Optional type for the value field. If None, inferred from
        value. If passed, must be one of str, int, float, bool, or json.
        Any other value will be treated as json (and will fail if the
        value can't be serialized to json).
    """
    __tablename__ = 'tags'

    file_path = Column(String, ForeignKey('files.path'), primary_key=True)
    entity_name = Column(String, ForeignKey('entities.name'), primary_key=True)
    _value = Column(String, nullable=False)
    _dtype = Column(String, default='str')

    file = relationship('BIDSFile', backref=backref(
        "tags", collection_class=attribute_mapped_collection("entity_name")))
    entity = relationship('Entity', backref=backref(
        "tags", collection_class=attribute_mapped_collection("file_path")))

    def __init__(self, file, entity, value, dtype=None):

        if dtype is None:
            dtype = type(value)

        self.value = value

        if not isinstance(dtype, str):
            dtype = dtype.__name__
        if dtype not in ('str', 'float', 'int', 'bool'):
            # Try serializing to JSON first
            try:
                value = json.dumps(value)
                dtype = 'json'
            except:
                raise ValueError(
                    "Passed value has an invalid dtype ({}). Must be one of "
                    "int, float, bool, or 'str.".format(dtype))
        value = str(value)
        self.file_path = file.path
        self.entity_name = entity.name

        self._value = value
        self._dtype = dtype

        self._init_on_load()

    def __repr__(self):
        msg = "<Tag file:{!r} entity:{!r} value:{!r}>"
        return msg.format(self.file_path, self.entity_name, self.value)

    @reconstructor
    def _init_on_load(self):
        if self._dtype not in ('str', 'float', 'int', 'bool', 'json'):
            raise ValueError("Invalid dtype '{}'. Must be one of 'int', "
                             "'float', 'bool', 'str', or 'json'.".format(self._dtype))
        if self._dtype == 'json':
            self.value = json.loads(self._value)
            self.dtype = 'json'
        else:
            self.dtype = eval(self._dtype)
            self.value = self.dtype(self._value)


class FileAssociation(Base):
    __tablename__ = 'associations'

    src = Column(String, ForeignKey('files.path'), primary_key=True)
    dst = Column(String, ForeignKey('files.path'), primary_key=True)
    kind = Column(String, primary_key=True)


# Association objects
config_to_entity_map = Table('config_to_entity_map', Base.metadata,
                             Column('config', String, ForeignKey('configs.name')),
                             Column('entity', String, ForeignKey('entities.name'))
                             )
