import argparse
import os
from pathlib import Path
from mne import (Info, setup_source_space, make_bem_model, make_bem_solution, make_forward_solution, Forward,
                 write_forward_solution)
from mne.io import read_info
from src.utils.file_access import load_json


def get_forward(info: Info, trans: str, subject: str, subjects_dir: Path, layers: int) -> Forward:
    """
    Get forward model specific for the subject. https://mne.tools/stable/auto_tutorials/forward/30_forward.html
    :param info: Info object about the data
    :param trans: Transform for the measurement
    :param subject: subject name
    :param subjects_dir: path to freesurfer directory
    :param layers: whether to use 1 or 3 layers, use one layer to avoid problems. For MEG 1 is enough.
    :return:
    """

    src = setup_source_space(subject, spacing="ico5", add_dist="patch", subjects_dir=subjects_dir)

    if layers == 3:
        conductivity = (0.3, 0.006, 0.3)    # for three layers
    elif layers == 1:
        conductivity = (0.3,)  # for single layer
    else:
        raise ValueError(f"Invalid layer number \"{layers}\" was given")

    model = make_bem_model(subject=subject, ico=5, conductivity=conductivity, subjects_dir=subjects_dir)
    bem = make_bem_solution(model)

    fwd = make_forward_solution(info=info, trans=trans, src=src, bem=bem,
                                meg=True, eeg=False, mindist=5.0, n_jobs=1,
                                verbose=True)
    return fwd


def get_args():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Create forward operator for the subject")
    parser.add_argument("--json_path", type=str, required=False, help="Path to JSON containing parameters")
    args = parser.parse_args()
    params = load_json(Path(args.json_path))

    info_path, trans_path, subject, subjects_dir, layers, dst_dir = params["info-path"], params["trans-path"], \
                                                                    params["subject"], params["subjects-dir"], \
                                                                    params["layers"], params["dst-dir"]

    # Convert to path objects
    subjects_dir = Path(subjects_dir)
    dst_dir = Path(dst_dir)

    # Make sure the directory exists
    if not dst_dir.exists():
        os.makedirs(dst_dir)

    return info_path, trans_path, subject, subjects_dir, layers, dst_dir


if __name__ == "__main__":
    info_path, trans_path, subject, subjects_dir, layers, dst_dir = get_args()
    info = read_info(info_path)
    fwd = get_forward(info, trans_path, subject, subjects_dir, layers)
    write_forward_solution(dst_dir / f"{subject}-fwd.fif", fwd, overwrite=True)