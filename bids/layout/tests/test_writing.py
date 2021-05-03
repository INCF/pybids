"""Tests related to file-writing functionality."""

import shutil
import os
from os.path import join, exists, islink, dirname

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bids.layout.writing import build_path, _PATTERN_FIND
from bids.tests import get_test_data_path
from bids import BIDSLayout
from bids.layout.models import BIDSFile, Entity, Tag, Base


@pytest.fixture
def writable_file(tmpdir):
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    testfile = 'sub-03_ses-2_task-rest_acq-fullbrain_run-2_bold.nii.gz'
    fn = tmpdir.mkdir("tmp").join(testfile)
    fn.write('###')
    bf = BIDSFile(os.path.join(str(fn)))

    tag_dict = {
        'task': 'rest',
        'run': 2,
        'subject': '3'
    }
    ents = {name: Entity(name) for name in tag_dict.keys()}
    tags = [Tag(bf, ents[k], value=v)
            for k, v in tag_dict.items()]

    session.add_all(list(ents.values()) + tags + [bf])
    session.commit()
    return bf


@pytest.fixture(scope='module')
def tmp_bids(tmpdir_factory):
    tmp_bids = tmpdir_factory.mktemp("tmp_bids")
    yield tmp_bids
    shutil.rmtree(str(tmp_bids))
    # Ugly hack
    try:
        shutil.rmtree(join(get_test_data_path(), '7t_trt', 'sub-Bob'))
    except:
        pass


@pytest.fixture(scope='module')
def layout(tmp_bids):
    orig_dir = join(get_test_data_path(), '7t_trt')
    # return BIDSLayout(data_dir, absolute_paths=False)
    new_dir = join(str(tmp_bids), 'bids')
    os.symlink(orig_dir, new_dir)
    return BIDSLayout(new_dir)


class TestWritableFile:

    def test_parse_pattern_re(self):
        """Unit tests on the strict entity pattern finder regex."""
        assert _PATTERN_FIND.findall('{extension<nii|nii.gz|json>|nii.gz}') == [
            ('{extension<nii|nii.gz|json>|nii.gz}', 'extension', 'nii|nii.gz|json', 'nii.gz')
        ]
        assert _PATTERN_FIND.findall('{extension<.nii|.nii.gz|.json>|.nii.gz}') == [
            ('{extension<.nii|.nii.gz|.json>|.nii.gz}', 'extension', '.nii|.nii.gz|.json', '.nii.gz')
        ]
        assert _PATTERN_FIND.findall('{extension<json|jsld>|json}') == [
            ('{extension<json|jsld>|json}', 'extension', 'json|jsld', 'json')
        ]
        assert _PATTERN_FIND.findall('{task<func|rest>}/r-{run}.nii.gz') == [
            ('{task<func|rest>}', 'task', 'func|rest', ''),
            ('{run}', 'run', '', '')
        ]

        pattern = """\
sub-{subject}[/ses-{session}]/anat/sub-{subject}[_ses-{session}][_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}]\
[_space-{space}]_{suffix<T1w|T2w|T1rho|T1map|T2map|T2star|FLAIR|FLASH|PDmap|PD|PDT2|inplaneT[12]|angio>}.\
{extension<nii|nii.gz|json>|nii.gz}"""
        assert sorted(_PATTERN_FIND.findall(pattern)) == [
            ('{acquisition}', 'acquisition', '', ''),
            ('{ceagent}', 'ceagent', '', ''),
            ('{extension<nii|nii.gz|json>|nii.gz}', 'extension', 'nii|nii.gz|json', 'nii.gz'),
            ('{reconstruction}', 'reconstruction', '', ''),
            ('{session}', 'session', '', ''),
            ('{session}', 'session', '', ''),
            ('{space}', 'space', '', ''),
            ('{subject}', 'subject', '', ''),
            ('{subject}', 'subject', '', ''),
            (
                '{suffix<T1w|T2w|T1rho|T1map|T2map|T2star|FLAIR|FLASH|PDmap|'
                'PD|PDT2|inplaneT[12]|angio>}',
                'suffix',
                'T1w|T2w|T1rho|T1map|T2map|T2star|FLAIR|FLASH|PDmap|PD|PDT2|inplaneT[12]|angio',
                ''
            )
        ]

    def test_build_path(self, writable_file):

        # Single simple pattern
        with pytest.raises(TypeError):
            build_path(writable_file.entities)
        pat = join(writable_file.dirname,
                   '{task}/sub-{subject}/run-{run}.nii.gz')
        target = join(writable_file.dirname, 'rest/sub-3/run-2.nii.gz')
        assert build_path(writable_file.entities, pat) == target

        # Multiple simple patterns
        pats = ['{session}/{task}/r-{run}.nii.gz',
                't-{task}/{subject}-{run}.nii.gz',
                '{subject}/{task}.nii.gz']
        pats = [join(writable_file.dirname, p) for p in pats]
        target = join(writable_file.dirname, 't-rest/3-2.nii.gz')
        assert build_path(writable_file.entities, pats) == target

        # Pattern with optional entity
        pats = ['[{session}/]{task}/r-{run}.nii.gz',
                't-{task}/{subject}-{run}.nii.gz']
        pats = [join(writable_file.dirname, p) for p in pats]
        target = join(writable_file.dirname, 'rest/r-2.nii.gz')
        assert build_path(writable_file.entities, pats) == target

        # Pattern with conditional values
        pats = ['{task<func|acq>}/r-{run}.nii.gz',
                't-{task}/{subject}-{run}.nii.gz']
        pats = [join(writable_file.dirname, p) for p in pats]
        target = join(writable_file.dirname, 't-rest/3-2.nii.gz')
        assert build_path(writable_file.entities, pats) == target

        # Pattern with valid conditional values
        pats = ['{task<func|rest>}/r-{run}.nii.gz',
                't-{task}/{subject}-{run}.nii.gz']
        pats = [join(writable_file.dirname, p) for p in pats]
        target = join(writable_file.dirname, 'rest/r-2.nii.gz')
        assert build_path(writable_file.entities, pats) == target

        # Pattern with optional entity with conditional values
        pats = ['[{task<func|acq>}/]r-{run}.nii.gz',
                't-{task}/{subject}-{run}.nii.gz']
        pats = [join(writable_file.dirname, p) for p in pats]
        target = join(writable_file.dirname, 'r-2.nii.gz')
        assert build_path(writable_file.entities, pats) == target

        # Pattern with default value
        pats = ['ses-{session|A}/r-{run}.nii.gz']
        assert build_path({'run': 3}, pats) == 'ses-A/r-3.nii.gz'

        # Pattern with both valid and default values
        pats = ['ses-{session<A|B|C|D>|D}/r-{run}.nii.gz']
        assert build_path({'run': 3}, pats) == 'ses-D/r-3.nii.gz'
        pats = ['ses-{session<A|B|C|D>|D}/r-{run}.nii.gz']
        assert build_path({'session': 'B', 'run': 3}, pats) == 'ses-B/r-3.nii.gz'

        # Test extensions with dot and warning is issued
        pats = ['ses-{session<A|B|C>|D}/r-{run}.{extension}']
        with pytest.warns(UserWarning) as record:
            assert build_path({'session': 'B', 'run': 3, 'extension': '.nii'},
                              pats) == 'ses-B/r-3.nii'
        assert "defines an invalid default value" in record[0].message.args[0]

        # Test expansion of optional characters
        pats = ['ses-{session<[ABCD]>|D}/r-{run}.{extension}']
        assert build_path({'session': 'B', 'run': 3, 'extension': '.nii'},
                          pats) == 'ses-B/r-3.nii'

        # Test default-only patterns are correctly overridden by setting entity
        entities = {
            'subject': '01',
            'extension': 'bvec',
            'suffix': 'T1rho',
        }
        pats = (
            "sub-{subject}[/ses-{session}]/{datatype|dwi}/sub-{subject}[_ses-{session}]"
            "[_acq-{acquisition}]_{suffix|dwi}.{extension<bval|bvec|json|nii.gz|nii>|nii.gz}"
        )
        assert build_path(entities, pats) == 'sub-01/dwi/sub-01_T1rho.bvec'
        assert build_path(entities, pats, strict=True) == 'sub-01/dwi/sub-01_T1rho.bvec'

        # Test multiple paths
        pats = ['ses-{session<A|B|C>|D}/r-{run}.{extension<json|nii|nii.gz>|nii.gz}']
        assert sorted(
            build_path({
                'session': ['A', 'B'],
                'run': [1, 2],
                'extension': ['.nii.gz', 'json']
            }, pats)) == [
            'ses-A/r-1.json',
            'ses-A/r-1.nii.gz',
            'ses-A/r-2.json',
            'ses-A/r-2.nii.gz',
            'ses-B/r-1.json',
            'ses-B/r-1.nii.gz',
            'ses-B/r-2.json',
            'ses-B/r-2.nii.gz',
        ]


    def test_strict_build_path(self):

        # Test with strict matching--should fail
        pats = ['[{session}/]{task}/r-{run}.nii.gz',
                't-{task}/{subject}-{run}.nii.gz']
        entities = {'subject': 1, 'task': "A", 'run': 2}
        assert build_path(entities, pats, True)
        entities = {'subject': 1, 'task': "A", 'age': 22}
        assert not build_path(entities, pats, True)

    def test_build_file(self, writable_file, tmp_bids, caplog):

        # Simple write out
        new_dir = join(writable_file.dirname, 'rest')
        pat = join(writable_file.dirname,
                   '{task}/sub-{subject}/run-{run}.nii.gz')
        target = join(writable_file.dirname, 'rest/sub-3/run-2.nii.gz')
        writable_file.copy(pat)
        assert exists(target)

        # Conflict handling
        with pytest.raises(ValueError):
            writable_file.copy(pat)
        with pytest.raises(ValueError):
            writable_file.copy(pat, conflicts='fail')
        with pytest.warns(UserWarning) as record:
            writable_file.copy(pat, conflicts='skip')
            log_message = record[0].message.args[0]
            assert log_message == 'A file at path {} already exists, ' \
                                  'skipping writing file.'.format(target)
        writable_file.copy(pat, conflicts='append')
        append_target = join(writable_file.dirname,
                             'rest/sub-3/run-2_1.nii.gz')
        assert exists(append_target)
        writable_file.copy(pat, conflicts='overwrite')
        assert exists(target)
        shutil.rmtree(new_dir)

        # Symbolic linking
        writable_file.copy(pat, symbolic_link=True)
        assert islink(target)
        shutil.rmtree(new_dir)

        # Using different root
        root = str(tmp_bids.mkdir('tmp2'))
        pat = join(root, '{task}/sub-{subject}/run-{run}.nii.gz')
        target = join(root, 'rest/sub-3/run-2.nii.gz')
        writable_file.copy(pat, root=root)
        assert exists(target)

        # Copy into directory functionality
        pat = join(writable_file.dirname, '{task}/')
        writable_file.copy(pat)
        target = join(writable_file.dirname, 'rest', writable_file.filename)
        assert exists(target)
        shutil.rmtree(new_dir)


class TestWritableLayout:

    def test_write_files(self, tmp_bids, layout):

        tmpdir = str(tmp_bids)
        pat = join(str(tmpdir), 'sub-{subject<02>}'
                                '/ses-{session}'
                                '/r-{run}'
                                '/suffix-{suffix}'
                                '/acq-{acquisition}'
                                '/task-{task}.nii.gz')
        layout.copy_files(path_patterns=pat)
        example_file = join(str(tmpdir), 'sub-02'
                                         '/ses-2'
                                         '/r-1'
                                         '/suffix-bold'
                                         '/acq-fullbrain'
                                         '/task-rest.nii.gz')
        example_file2 = join(str(tmpdir), 'sub-01'
                                          '/ses-2'
                                          '/r-1'
                                          '/suffix-bold'
                                          '/acq-fullbrain'
                                          '/task-rest.nii.gz')

        assert exists(example_file)
        assert not exists(example_file2)

        pat = join(str(tmpdir), 'sub-{subject<01>}'
                                '/ses-{session}'
                                '/r-{run}'
                                '/suffix-{suffix}'
                                '/task-{task}.nii.gz')
        example_file = join(str(tmpdir), 'sub-01'
                                         '/ses-2'
                                         '/r-1'
                                         '/suffix-bold'
                                         '/task-rest.nii.gz')
        # Should fail without the 'overwrite' because there are multiple
        # files that produce the same path.
        with pytest.raises(ValueError):
            layout.copy_files(path_patterns=pat)
        try:
            os.remove(example_file)
        except OSError:
            pass
        layout.copy_files(path_patterns=pat, conflicts='overwrite')
        assert exists(example_file)

    def test_write_to_file(self, tmp_bids, layout):
        contents = 'test'
        entities = {'subject': 'Bob', 'session': '01'}
        pat = join('sub-{subject}/ses-{session}/desc.txt')
        layout.write_to_file(entities, path_patterns=pat,
                                      contents=contents, validate=False)
        target = join(str(tmp_bids), 'bids', 'sub-Bob/ses-01/desc.txt')
        assert exists(target)
        with open(target) as f:
            written = f.read()
        assert written == contents
        assert target not in layout.files

    def test_write_to_file_defaults(self, tmp_bids, layout):
        contents = 'test'
        entities = {'subject': 'Bob', 'session': '01', 'run': '1',
                    'suffix': 'bold', 'task': 'test', 'acquisition': 'test',
                    'bval': 0}
        layout.write_to_file(entities, contents=contents)
        target = join(str(tmp_bids), 'bids', 'sub-Bob', 'ses-01',
                      'func', 'sub-Bob_ses-01_task-test_acq-test_run-1_bold.nii.gz')
        assert exists(target)
        with open(target) as f:
            written = f.read()
        assert written == contents

    def test_build_file_from_layout(self, tmpdir, layout):
        entities = {'subject': 'Bob', 'session': '01', 'run': '1'}
        pat = join(str(tmpdir), 'sub-{subject}'
                   '/ses-{session}'
                   '/r-{run}.nii.gz')
        path = layout.build_path(entities, path_patterns=pat, validate=False)
        assert path == join(str(tmpdir), 'sub-Bob/ses-01/r-1.nii.gz')

        data_dir = join(dirname(__file__), 'data', '7t_trt')
        filename = 'sub-04_ses-1_task-rest_acq-fullbrain_run-1_physio.tsv.gz'
        file = join('sub-04', 'ses-1', 'func', filename)
        path = layout.build_path(file, path_patterns=pat, validate=False)
        assert path.endswith('sub-04/ses-1/r-1.nii.gz')
