"""Functions for building strings for individual parameters.
"""
import logging
import os
import os.path as op

import math
import nibabel as nib
from num2words import num2words

from .utils import num_to_str, list_to_str, remove_duplicates

logging.basicConfig()
LOGGER = logging.getLogger("pybids.reports.parsing")


def get_slice_str(img, metadata):
    if "SliceTiming" in metadata.keys():
        slice_order = " in {0} order".format(get_slice_info(metadata["SliceTiming"]))
        n_slices = len(metadata["SliceTiming"])
    else:
        slice_order = ""
        n_slices = img.shape[2]
    slice_str = "{n_slices} slices{slice_order}".format(
        n_slices=n_slices, slice_order=slice_order
    )
    return slice_str


def get_tr_str(metadata):
    tr = metadata["RepetitionTime"] * 1000
    tr = num_to_str(tr)
    tr_str = "repetition time, TR={tr}ms".format(tr=tr)
    return tr_str


def get_func_duration(n_vols, tr):
    run_secs = math.ceil(n_vols * tr)
    mins, secs = divmod(run_secs, 60)
    duration = "{0}:{1:02.0f}".format(int(mins), int(secs))
    return duration


def get_dur_str(files):
    first_file = files[0]
    metadata = first_file.get_metadata()
    tr = metadata["RepetitionTime"]
    imgs = [nib.load(f) for f in files]
    n_vols = [img.shape[3] for img in imgs]
    if len(set(n_vols)) > 1:
        min_vols = min(n_vols)
        max_vols = max(n_vols)
        min_dur = get_func_duration(min_vols, tr)
        max_dur = get_func_duration(max_vols, tr)
        dur_str = "{}-{}".format(min_dur, max_dur)
        n_vols = "{}-{}".format(min_vols, max_vols)
    else:
        n_vols = n_vols[0]
        dur_str = get_func_duration(n_vols, tr)

    dur_str = (
        "Run duration was {0} minutes, during which {1} volumes were acquired."
    ).format(dur_str, n_vols)
    return dur_str


def get_mbfactor_str(metadata):
    """Build a description of the multi-band acceleration applied, if used."""
    if metadata.get("MultibandAccelerationFactor", 1) > 1:
        mb_str = "MB factor={}".format(metadata["MultibandAccelerationFactor"])
    else:
        mb_str = ""
    return mb_str


def get_echotimes_str(files):
    """Build a description of echo times from metadata field.

    Parameters
    ----------
    metadata : dict
        Metadata information for multiple files merged into one dictionary.
        For multi-echo data, EchoTime should be a list.

    Returns
    -------
    te_str : str
        Description of echo times.
    me_str : str
        Whether the data are multi-echo or single-echo.
    """
    echo_times = [f.get_metadata()["EchoTime"] for f in files]
    echo_times = sorted(list(set(echo_times)))
    if len(echo_times) > 1:
        te = [num_to_str(t * 1000) for t in echo_times]
        te = list_to_str(te)
        me_str = "multi-echo"
    else:
        te = num_to_str(echo_times[0] * 1000)
        me_str = "single-echo"
    te_str = "echo time, TE={}ms".format(te)
    return te_str, me_str


def get_size_strs(img):
    """Build descriptions from sizes of imaging data, including field of view,
    voxel size, and matrix size.

    Parameters
    ----------
    img : nibabel.nifti1.Nifti1Image
        Image object from which to determine sizes.

    Returns
    -------
    fov_str
    matrixsize_str
    voxelsize_str
    """
    vs_str, ms_str, fov_str = get_sizestr(img)
    fov_str = "field of view, FOV={}mm".format(fov_str)
    voxelsize_str = "voxel size={}mm".format(vs_str)
    matrixsize_str = "matrix size={}".format(ms_str)
    return fov_str, matrixsize_str, voxelsize_str


def get_inplaneaccel_str(metadata):
    if metadata.get("ParallelReductionFactorInPlane", 1) > 1:
        pr_str = "in-plane acceleration factor={}".format(
            metadata["ParallelReductionFactorInPlane"]
        )
    else:
        pr_str = ""
    return pr_str


def get_flipangle_str(metadata):
    return "flip angle, FA={}<deg>".format(metadata.get("FlipAngle", "UNKNOWN"))


def get_nvecs_str(img):
    return "{} diffusion directions".format(img.shape[3])


def get_bval_str(bval_file):
    # Parse bval file
    with open(bval_file, "r") as file_object:
        d = file_object.read().splitlines()
    bvals = [item for sublist in [l.split(" ") for l in d] for item in sublist]
    bvals = sorted([int(v) for v in set(bvals)])
    bvals = [num_to_str(v) for v in bvals]
    bval_str = list_to_str(bvals)
    bval_str = "b-values of {} acquired".format(bval_str)
    return bval_str


def get_dir_str(metadata, config):
    dir_str = config["dir"][metadata["PhaseEncodingDirection"]]
    dir_str = "phase encoding: {}".format(dir_str)
    return dir_str


def get_for_str(metadata, layout):
    if "IntendedFor" in metadata.keys():
        scans = metadata["IntendedFor"]
        run_dict = {}
        for scan in scans:
            fn = op.basename(scan)
            if_file = [
                f for f in layout.get(extension=[".nii", ".nii.gz"]) if fn in f.path
            ][0]
            run_num = int(if_file.run)
            target_type = if_file.entities["suffix"].upper()
            if target_type == "BOLD":
                iff_meta = layout.get_metadata(if_file.path)
                task = iff_meta.get("TaskName", if_file.entities["task"])
                target_type_str = "{0} {1} scan".format(task, target_type)
            else:
                target_type_str = "{0} scan".format(target_type)

            if target_type_str not in run_dict.keys():
                run_dict[target_type_str] = []
            run_dict[target_type_str].append(run_num)

        for scan in run_dict.keys():
            run_dict[scan] = [
                num2words(r, ordinal=True) for r in sorted(run_dict[scan])
            ]

        out_list = []
        for scan in run_dict.keys():
            if len(run_dict[scan]) > 1:
                s = "s"
            else:
                s = ""
            run_str = list_to_str(run_dict[scan])
            string = "{rs} run{s} of the {sc}".format(rs=run_str, s=s, sc=scan)
            out_list.append(string)
        for_str = " for the {0}".format(list_to_str(out_list))
    else:
        for_str = ""
    return for_str


def get_slice_info(slice_times):
    """
    Extract slice order from slice timing info.

    TODO: Be more specific with slice orders.
    Currently anything where there's some kind of skipping is interpreted as
    interleaved of some kind.

    Parameters
    ----------
    slice_times : array-like
        A list of slice times in seconds or milliseconds or whatever.

    Returns
    -------
    slice_order_name : :obj:`str`
        The name of the slice order sequence.
    """
    # Slice order
    slice_times = remove_duplicates(slice_times)
    slice_order = sorted(range(len(slice_times)), key=lambda k: slice_times[k])
    if slice_order == range(len(slice_order)):
        slice_order_name = "sequential ascending"
    elif slice_order == reversed(range(len(slice_order))):
        slice_order_name = "sequential descending"
    elif slice_order[0] < slice_order[1]:
        # We're allowing some wiggle room on interleaved.
        slice_order_name = "interleaved ascending"
    elif slice_order[0] > slice_order[1]:
        slice_order_name = "interleaved descending"
    else:
        slice_order = [str(s) for s in slice_order]
        raise Exception("Unknown slice order: [{0}]".format(", ".join(slice_order)))

    return slice_order_name


def get_seqstr(metadata, config):
    """
    Extract and reformat imaging sequence(s) and variant(s) into pretty
    strings.

    Parameters
    ----------
    metadata : :obj:`dict`
        The metadata for the scan.
    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    seqs : :obj:`str`
        Sequence names.
    variants : :obj:`str`
        Sequence variant names.
    """
    seq_abbrs = metadata.get("ScanningSequence", "").split("_")
    seqs = [config["seq"].get(seq, seq) for seq in seq_abbrs]
    variants = [
        config["seqvar"].get(var, var)
        for var in metadata.get("SequenceVariant", "").split("_")
    ]
    seqs = list_to_str(seqs)
    if seq_abbrs[0]:
        seqs += " ({0})".format(os.path.sep.join(seq_abbrs))
    variants = list_to_str(variants)
    return seqs, variants


def get_sizestr(img):
    """
    Extract and reformat voxel size, matrix size, field of view, and number of
    slices into pretty strings.

    Parameters
    ----------
    img : :obj:`nibabel.Nifti1Image`
        Image from scan from which to derive parameters.

    Returns
    -------
    voxel_size : :obj:`str`
        Voxel size string (e.g., '2x2x2')
    matrix_size : :obj:`str`
        Matrix size string (e.g., '128x128')
    fov : :obj:`str`
        Field of view string (e.g., '256x256')
    """
    n_x, n_y = img.shape[:2]
    import numpy as np

    voxel_dims = np.array(img.header.get_zooms()[:3])
    matrix_size = "{0}x{1}".format(num_to_str(n_x), num_to_str(n_y))
    voxel_size = "x".join([num_to_str(s) for s in voxel_dims])
    fov = [n_x, n_y] * voxel_dims[:2]
    fov = "x".join([num_to_str(s) for s in fov])
    return voxel_size, matrix_size, fov
