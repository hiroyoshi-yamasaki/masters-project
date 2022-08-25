import argparse
import logging
import os
import re
import sys
import traceback

from pathlib import Path

import numpy as np

from joblib import Parallel, delayed
from mne import (compute_covariance, read_labels_from_annot, Epochs, Evoked, Label, read_forward_solution,
                 read_source_spaces, compute_source_morph, read_epochs)
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs

from src.utils.exceptions import SubjectNotProcessedError
from src.utils.file_access import load_json
from src.utils.logger import get_logger

logger = get_logger(file_name="source-localization")
logger.setLevel(logging.INFO)


########################################################################################################################
# SOURCE LOCALIZATION                                                                                                  #
########################################################################################################################


def source_localize(dst_dir: Path, subject: str, epochs: Epochs, params: dict, n_jobs=1) -> None:
    """
    Source localize the epoch data and save the results
    :param dst_dir: path to directory to save the results in
    :param subject: name of the subject
    :param epochs: Epochs object to perform source localization on
    :param params: parameter dictionary
    :param n_jobs: number of jobs for parallelism
    """

    logger.info(f"Source localizing {subject} files")

    # Make inverse model
    logger.info(f"Making an inverse model for the subject {subject} ")
    inv = get_inv(epochs, fwd_path=str(Path(params["fwd-dir"]) / f"{subject}-fwd.fif"), n_jobs=n_jobs)

    # Common source space
    logger.info(f"Setting up morph to FS average")
    fsaverage_src_path = Path(params["subjects-dir"]) / "fsaverage" / "bem" / "fsaverage-ico-5-src.fif"
    fs_src = read_source_spaces(str(fsaverage_src_path))
    morph = compute_source_morph(src=inv["src"], subject_from=subject, subject_to="fsaverage", src_to=fs_src,
                                 subjects_dir=params["subjects-dir"], verbose=False)

    # Generate set of labels
    logger.info(f"Reading labels")
    labels = read_labels_from_annot("fsaverage", params["parcellation"], params["hemi"],
                                    subjects_dir=params["subjects-dir"], verbose=False)

    # Create parallel functions per time point
    parallel_funcs = []

    for label in labels:
        # Ignore irrelevant labels
        if re.match(r".*(unknown|\?|deeper|cluster|default|ongur|medial\.wall).*", label.name.lower()):
            continue
        func = delayed(_process_single_label)(dst_dir=dst_dir,
                                              epochs=epochs, label=label, inv=inv,
                                              params=params, morph=morph)
        parallel_funcs.append(func)

    logger.info(f"Total of {len(parallel_funcs)} parallel functions added")
    logger.info(f"Executing {n_jobs} jobs in parallel")
    parallel_pool = Parallel(n_jobs=n_jobs)
    parallel_pool(parallel_funcs)

    logger.debug(f"{len(parallel_funcs)} time steps processed")


def _process_single_label(dst_dir: Path, epochs: Epochs, label: Label, inv, params, morph) -> None:
    """
    Perform source localization on a particular cortical area.
    :param dst_dir: directory to store the results in
    :param epochs: epochs object to perform source localization on
    :param label: cortical area of interest
    :param inv: inverse operator
    :param params: should contain `method` and `pick ori`
    :param morph: todo morph
    """

    logger.info(f"Processing single subject for {label.name} ")

    stcs = _inverse_epochs(epochs, inv=inv, method=params["method"], pick_ori=params["pick-ori"])
    stcs = _morph_to_common(stcs, morph)

    data_list = []
    for stc in stcs:
        stc = stc.in_label(label)
        data_list.append(stc.data)

    data = np.stack(data_list)

    _write_array(dst_dir=dst_dir, label=label, data_array=data)


def _morph_to_common(stcs: list, morph):
    """
    Morph the source localization into a common (fsaverge) space
    :param stcs: list of source localizations (per epoch)
    :param morph: todo morph
    :return: todo return
    """

    logger.info(f"Morphing to fsaverage")

    for stc in stcs:
        fs_stc = morph.apply(stc)
        yield fs_stc


def _write_array(dst_dir: Path, label: Label, data_array) -> None:
    """
    Write the resulting file into appropriate directory
    :param dst_dir: path to directory in which results will be saved
    :param label: name of the cortical area
    :param data_array: data array to be saved
    """

    logger.info(f"Writing the data for {label.name} to file")

    stc_fname = f"{label.name}.npy"

    logger.info(f"Saving {stc_fname} to file in {dst_dir}")

    stc_dir = dst_dir / "stc"

    if not stc_dir.exists():
        os.makedirs(stc_dir)

    try:
        np.save(str(stc_dir / stc_fname), data_array)

    except OSError as e:
        logger.exception(f"Failed to write {dst_dir / stc_fname} to file. {e.strerror}")
        raise SubjectNotProcessedError(e)


def _inverse_evoked(evoked: Evoked, fwd_path: str, method="dSPM", snr=3., return_residual=True, pick_ori=None, inv=None,
                    epochs=None, n_jobs=1, tmax=0.,
                    inv_method=("shrunk", "empirical"), rank=None,
                    loose=0.2, depth=0.8, verbose=False):
    """
    todo comment
    :param evoked: evoked object
    :param fwd_path: path to precomputed forward object
    :param method:
    :param snr: signal to noise ratio
    :param return_residual: return residual (see MNE)
    :param pick_ori: pick orientation (see MNE)
    :param inv: inverse operator object
    :param epochs: epochs object
    :param n_jobs: number of jobs
    :param tmax: todo tmax
    :param inv_method: source estimation method
    :param rank: todo rank
    :param loose: todo
    :param depth: todo
    :param verbose: verbose
    :return:
        source estimation
    """

    if not inv:
        inv = get_inv(epochs, fwd_path=fwd_path, n_jobs=n_jobs, tmax=tmax, method=inv_method, rank=rank,
                      loose=loose, depth=depth, verbose=verbose)

    lambda2 = 1. / snr ** 2
    return apply_inverse(evoked, inv, lambda2,
                         method=method, pick_ori=pick_ori,
                         return_residual=return_residual, verbose=False)


def _inverse_epochs(epochs: Epochs, label=None, method="dSPM", snr=3., pick_ori=None, inv=None,
                    n_jobs=1, tmax=0., fwd_path="",
                    inv_method=("shrunk", "empirical"), rank=None,
                    loose=0.2, depth=0.8, verbose=False):
    """
    todo comment
    :param epochs: epochs object
    :param label: labels (cortical area)
    :param method: source estimation method
    :param snr: signal to noise ratio
    :param pick_ori: todo pick ori
    :param inv: inverse operator
    :param n_jobs: number of jobs
    :param tmax: todo tmax
    :param fwd_path: path to precomputed forward operator
    :param inv_method: todo inv_method
    :param rank: todo rank
    :param loose: todo loose
    :param depth: todo depth
    :param verbose: verbosity
    :return:
        source estimation
    """

    logger.info(f"Inverting epochs")

    if not inv:
        inv = get_inv(epochs, fwd_path=fwd_path, n_jobs=n_jobs, tmax=tmax, method=inv_method, rank=rank,
                      loose=loose, depth=depth, verbose=verbose)

    lambda2 = 1. / snr ** 2
    return apply_inverse_epochs(epochs, inv, lambda2, label=label,
                                method=method, pick_ori=pick_ori, verbose=verbose, return_generator=True)


def get_inv(epochs: Epochs, fwd_path: str, tmax=0., n_jobs=1, method=("shrunk", "empirical"),
            rank=None, loose=0.2, depth=0.8, verbose=False):
    """
    todo comment
    :param epochs: epochs object
    :param fwd_path: path to precomputed forward operator
    :param tmax: todo tmax
    :param n_jobs: number of jobs for parallelism
    :param method: todo method
    :param rank: todo rank
    :param loose: todo loose
    :param depth: todo depth
    :param verbose: verbosity
    :return:
        inverse operator
    """

    fwd = read_forward_solution(fwd_path, verbose=verbose)
    noise_cov = compute_covariance(epochs, tmax=tmax, method=method, rank=rank, n_jobs=n_jobs, verbose=verbose)
    inv = make_inverse_operator(epochs.info, fwd, noise_cov, loose=loose, depth=depth, verbose=verbose)

    return inv


def get_labels_names(params: dict):
    """
    todo comment
    :param params: todo params
    :return: todo return
    """

    # Generate set of labels
    labels = read_labels_from_annot("fsaverage", params["parcellation"], params["hemi"],
                                    subjects_dir=params["subjects dir"], verbose=False)
    label_names = []
    for label in labels:

        # Ignore irrelevant labels
        if not re.match(r".*(unknown|\?|deeper|cluster|default|ongur|medial\.wall).*", label.name.lower()):

            label_names.append(label.name)

    return label_names


def get_args():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Epoch raw files")

    # Read parameter from JSON
    parser.add_argument("--json_path", type=str, required=False, help="Path to JSON containing parameters")

    # Add arguments
    parser.add_argument("--dst_dir", type=str, required=False, help="Directory to save the results in")
    parser.add_argument("--subject", type=str, required=False, help="Name of the subject. e.g. `sub-V1001`")
    parser.add_argument("--epochs_path", type=str, required=False, help="Path to epochs data")
    parser.add_argument("--parc", type=str, required=False, default="aparc",
                        help="Parcellation scheme to use. Default is `aparc`")
    parser.add_argument("--subjects_dir", type=str, required=False, help="Path to subjects_dir")
    parser.add_argument("--hemi", type=str, required=False, help="Hemisphere, `lh`, `rh` or `both`")
    parser.add_argument("--fwd_dir", type=str, required=False, help="Directory containing all forward models")
    # todo method, pickori
    args = parser.parse_args()

    # Either from JSON or one by one
    if args.json_path:
        params = load_json(args.json_path)
        dst_dir, subject, epochs_path = params["dst-dir"], params["subject"], params["epochs-path"]
        parc, subjects_dir, hemi, fwd_dir = params["parc"], params["subjects-dir"], params["hemi"], params["fwd-dir"]
        method, pick_ori = params["method"], params["pick-ori"]
    else:
        dst_dir, subject, epochs_path, parc, subjects_dir, hemi, fwd_dir, method, pick_ori = \
            args.dst_dir, args.subject, args.epochs_path, args.parc, args.subjects_dir, args.hemi, args.fwd_dir, args.method, args.pick_ori

    # Convert to Path object
    dst_dir = Path(dst_dir)
    epochs_path = Path(epochs_path)
    subjects_dir = Path(subjects_dir)

    if not dst_dir.exists():
        os.makedirs(dst_dir)

    # Convert to appropriate format
    params = {"parcellation": parc, "hemi": hemi,
              "subjects-dir": subjects_dir, "fwd-dir": fwd_dir,
              "method": method, "pick-ori": pick_ori}

    return dst_dir, subject, epochs_path, params


if __name__ == "__main__":

    try:
        # Read parameters
        dst_dir, subject, epochs_path, params = get_args()

        # Read epochs
        epochs = read_epochs(epochs_path)

        # Source localize
        source_localize(dst_dir=dst_dir, subject=subject, epochs=epochs, params=params)

    except FileNotFoundError as e:
        logger.exception(e.strerror)
        sys.exit(-1)

    except Exception as e:  # noqa

        logger.error(f"Unexpected exception during source localization. \n {traceback.format_exc()}")
        sys.exit(-1)

    logger.info(f"Source localization has finished.")
