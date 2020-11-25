""" Utility functions. """

import re
import os
from frozendict import frozendict


# Monkeypatch to print out frozendicts *as if* they were dictionaries.
frozendict.__repr__ = lambda s: repr({k: v for k, v in s.items()})


def listify(obj):
    ''' Wraps all non-list or tuple objects in a list; provides a simple way
    to accept flexible arguments. '''
    return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def hashablefy(obj):
    ''' Make dictionaries and lists hashable or raise. '''
    if isinstance(obj, list):
        return tuple([hashablefy(o) for o in obj])

    if isinstance(obj, dict):
        return frozendict({k: hashablefy(v) for k, v in obj.items()})
    return obj


def matches_entities(obj, entities, strict=False):
    ''' Checks whether an object's entities match the input. '''
    if strict and set(obj.entities.keys()) != set(entities.keys()):
        return False

    comm_ents = list(set(obj.entities.keys()) & set(entities.keys()))
    for k in comm_ents:
        current = obj.entities[k]
        target = entities[k]
        if isinstance(target, (list, tuple)):
            if current not in target:
                return False
        elif current != target:
            return False
    return True


def natural_sort(l, field=None):
    '''
    based on snippet found at http://stackoverflow.com/a/4836734/2445984
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        if field is not None:
            key = getattr(key, field)
        if not isinstance(key, str):
            key = str(key)
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def convert_JSON(j):
    """ Recursively convert CamelCase keys to snake_case.
    From: https://stackoverflow.com/questions/17156078/
    converting-identifier-naming-between-camelcase-and-
    underscores-during-json-seria
    """
    def camel_to_snake(s):
        a = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')
        return a.sub(r'_\1', s).lower()

    def convertArray(a):
        newArr = []
        for i in a:
            if isinstance(i, list):
                newArr.append(convertArray(i))
            elif isinstance(i, dict):
                newArr.append(convert_JSON(i))
            else:
                newArr.append(i)
        return newArr

    out = {}
    for k, value in j.items():
        newK = camel_to_snake(k)

        # Replace transformation uses a dict, so skip lower-casing
        if isinstance(value, dict) and k != 'Replace':
            out[newK] = convert_JSON(value)
        elif isinstance(value, list):
            out[newK] = convertArray(value)
        else:
            out[newK] = value

    return out


def splitext(path):
    """splitext for paths with directories that may contain dots.
    From https://stackoverflow.com/questions/5930036/separating-file-extensions-using-python-os-path-module"""
    li = []
    path_without_extensions = os.path.join(os.path.dirname(path),
        os.path.basename(path).split(os.extsep)[0])
    extensions = os.path.basename(path).split(os.extsep)[1:]
    li.append(path_without_extensions)
    # li.append(extensions) if you want extensions in another list inside the list that is returned.
    li.extend(extensions)
    return li


def make_bidsfile(filename):
    """Create a BIDSFile instance of the appropriate class. """
    from .layout import models

    patt = re.compile("[._]*[a-zA-Z0-9]*?(\\.[^/\\\\]+)$")
    m = re.search(patt, filename)

    ext = '' if not m else m.group(1)

    if ext.endswith(('.nii', '.nii.gz', '.gii')):
        cls = 'BIDSImageFile'
    elif ext in ['.tsv', '.tsv.gz']:
        cls = 'BIDSDataFile'
    elif ext == '.json':
        cls = 'BIDSJSONFile'
    else:
        cls = 'BIDSFile'

    Cls = getattr(models, cls)
    return Cls(filename)


def validate_multiple(val, retval=None):
    """Any click.Option with the multiple flag will return an empty tuple if not set.

    This helper method converts empty tuples to a desired return value (default: None).
    This helper method selects the first item in single-item tuples.
    """
    assert isinstance(val, tuple)

    if val == tuple():
        return retval
    if len(val) == 1:
        return val[0]
    return val
