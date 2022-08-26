import argparse
import os
from pathlib import Path
from mne.coreg import Coregistration
from mne import Info, Transform, write_trans
from mne.io import read_info
from src.utils.file_access import load_json

########################################################################################################################
# COREGISTRATION                                                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #
# Handles coregistration calculation for MEG data. Subject specific forward models are generated from coregistraion.   #
# Details about coregistration is found here: https://mne.tools/stable/auto_tutorials/forward/25_automated_coreg.html  #
########################################################################################################################


def get_trans(subject: str, subjects_dir: Path, info: Info) -> Transform:
    """
    Get Transform instance from the movement measurements.
    https://mne.tools/stable/auto_tutorials/forward/25_automated_coreg.html
    :param subject: name of the subject
    :param subjects_dir: freesurfer subject directory
    :param info: measurement info to be fed to the algorithm
    :return: Transform data
    """

    coreg = Coregistration(info, subject=subject, subjects_dir=subjects_dir, fiducials="auto")
    coreg.fit_icp(n_iterations=6, nasion_weight=2.)
    coreg.omit_head_shape_points(distance=5. / 1000)
    coreg.fit_icp(n_iterations=20, nasion_weight=10.)
    return coreg.trans


def get_args():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Create trans automatically for the subject")
    parser.add_argument("--json_path", type=str, required=False, help="Path to JSON containing parameters")
    args = parser.parse_args()
    params = load_json(Path(args.json_path))

    subject, subjects_dir, info_path, dst_dir = params["subject"], params["subjects-dir"], params["info-path"], \
                                                params["dst-dir"]

    # Convert to path objects
    subjects_dir = Path(subjects_dir)
    dst_dir = Path(dst_dir)

    # Make sure the directory exists
    if not dst_dir.exists():
        os.makedirs(dst_dir)

    return subject, subjects_dir, info_path, dst_dir


if __name__ == "__main__":
    subject, subjects_dir, info_path, dst_dir = get_args()
    info = read_info(info_path)
    trans = get_trans(subject, subjects_dir, info)
    write_trans(dst_dir, trans, overwrite=True)
