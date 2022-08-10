import logging
import re
import os
from pathlib import Path
from typing import List
import numpy as np

from src.utils.file_access import write_json
from src.utils.logger import get_logger

logger = get_logger(file_name="artifact")
logger.setLevel(logging.INFO)

# todo big comment at the top

########################################################################################################################
# DATASET GENERATION                                                                                                   #
########################################################################################################################


def _get_events_paths(epoch_dir: Path, reject_list: List[str]):
    """
    Collect list of available events arrays
    :param epoch_dir: directory containing all epochs
    :param reject_list: todo comment reject_list
    :return:
        list of paths to events array
    """

    fname = "events.npy"
    events_path_list = []

    for subject_dir in os.listdir(epoch_dir):   # e.g. "sub-V1001"

        # Skip rejected subjects
        if subject_dir in reject_list:
            continue

        subject_path = epoch_dir / subject_dir  # "epochs/sub-V1001"

        if re.match(r"^sub-[AV]\d+$", str(subject_dir)):  # there are hidden files in the directory
            events_path = subject_path / fname
            if events_path.exists():
                events_path_list.append(events_path)
            else:
                logger.debug(f"events file not found in {subject_dir}. Skipping...")
    logger.info(f"Found {len(events_path_list)}  events found")
    return events_path_list


def _get_stc_paths(epoch_dir: Path, area_name: str, reject_list: List[str]):
    """
    ollect list of available source localizations
    :param epoch_dir: directory containing all epochs
    :param area_name: name of the area
    :param reject_list: todo reject_list
    :return:
        list of source localization
    """

    stc_path_list = []

    for subject_dir in os.listdir(epoch_dir):   # e.g. "sub-V1001"

        # Skip rejected subjects
        if subject_dir in reject_list:
            continue

        subject_path = epoch_dir / subject_dir  # "epochs/sub-V1001"

        if re.match(r"^sub-[AV]\d+$", str(subject_dir)):  # there are hidden files in the directory

            # Look for matching cortical area
            stc_dir = subject_path / "stc"                      # e.g. "epochs/sub-V1001/stc"
            if stc_dir.exists():
                for area_file in os.listdir(stc_dir):               # e.g. "epochs/sub-V1001/stc/..."
                    if area_file.startswith(area_name):             # e.g. "fusiform_1-h.npy"
                        stc_path_list.append(stc_dir / area_file)   # e.g. "epochs/sub-V1001/stc/fusiform_1-h.npy"
            else:
                logger.info(f"No source reconstruction data for {subject_dir} available. Skipping...")

    logger.info(f"Found {len(stc_path_list)} source reconstruction files found")
    return stc_path_list


def _validate_paths(stc_paths: list, events_paths: list):
    """
    Validate the paths by making sure they are of the same subjectt
    :param stc_paths: list of paths to stc data
    :param events_paths: list of paths to events array
    :return: validated lists of paths
    """

    logger.info(f"Validating the paths")

    valid_stcs, valid_events = [], []
    for stc_path in stc_paths:
        subject = re.findall(r"sub-[AV]\d+", str(stc_path))[0]

        for event_path in events_paths:

            if re.search(rf"\.*{subject}\.*", str(event_path)):
                valid_stcs.append(stc_path)
                valid_events.append(event_path)
                break

    logger.info(f"{len(valid_stcs)} valid subject data found")
    return valid_stcs, valid_events


def _generate_mmap(dst_dir: Path, data_paths: List[Path], event_paths):
    # todo

    # Get the size of final array
    x_shape = _get_array_size(data_paths)
    fname = "x.dat"

    x_map = np.memmap(str(dst_dir / fname), dtype="float64", mode="w+", shape=x_shape)

    y_list = []

    # Use memory map for x
    curr_idx = 0
    for idx, (data_path, event_path) in enumerate(zip(data_paths, event_paths)):
        logger.debug(f"{idx} / {len(data_paths)}")
        logger.debug(f"Appending {data_path}")

        # Read x
        x = np.load(str(data_path))

        # Read y
        events = np.load(str(event_path))
        y = events[:, 2]

        # Make sure the shape stays the same
        if x.shape[0] == y.shape[0]:
            x_map[curr_idx: curr_idx + x.shape[0]] = x
            curr_idx += x.shape[0]
            x_map.flush()

            y_list.append(y)
        else:
            raise ValueError(f"The numbers of epochs for x {x.shape[0]} and y {y.shape[0]} are different")

    shape = {"shape": x_shape}
    fname = "x_shape.json"
    write_json(dst_dir, file_name=fname, data=shape)  # needed to recover the shape

    y = np.hstack(y_list)
    fname = "y.npy"
    np.save(str(dst_dir / fname), y)


def _get_array_size(paths):
    # todo comment

    dim_0 = 0
    dim_rest = None
    for path in paths:

        # Read single x data
        x = np.load(str(path))
        shape = x.shape
        del x

        dim_0 += shape[0]
        if dim_rest is None:
            dim_rest = shape[1:]

    x_shape = [dim_0]
    x_shape.extend(dim_rest)

    return tuple(x_shape)


def _get_reject_list(reject_path: Path):
    # todo comment

    with open(reject_path, "r") as file:
        reject_text = file.read()

    return reject_text.splitlines()


def _generate_data(dst_dir: Path, data_paths: list, event_paths: list):
    """
    Generate data by concatenating all arrays
    :param dst_dir: path to directory to store the results
    :param data_paths: paths the source localization data
    :param event_paths: paths to events arrays
    """

    x_list = []
    y_list = []

    added = 0
    for idx, (data_path, event_path) in enumerate(zip(data_paths, event_paths)):
        logger.debug(f"{idx} / {len(data_paths)}")
        logger.debug(f"Appending {data_path}")

        # Read x
        x = np.load(str(data_path))

        # Read y
        events = np.load(str(event_path))
        y = events[:, 2]

        # Make sure the shape stays the same
        if x.shape[0] == y.shape[0]:
            x_list.append(x)
            y_list.append(y)
            added += 1
        else:
            logger.debug(f"Number of events don’t match, skipping")

    # todo: add exception
    x = np.vstack(x_list)
    y = np.hstack(y_list)
    fname_x = "x.npy"
    fname_y = "y.npy"
    np.save(str(dst_dir / fname_x), x)
    np.save(str(dst_dir / fname_y), y)
    logger.debug(f"{added} subject data added")


def generate_dataset(epoch_dir: Path, dst_dir: Path, area_name: str, max_subjects=-1,
                     memmap=True, reject=None) -> None:
    """
    :param epoch_dir: directory in which stc and epochs are stored
    :param dst_dir: directory in which to save the results
    :param area_name: name of the cortical area
    :param memmap: if true, use memmap to store the results
    :param max_subjects: maximum number of subjects to include, if negative use all available data
    :param reject: allows specific to reject subjects
    :return:
    """

    logger.debug(f"Generating dataset for {area_name}")

    # Get list of subjects to ignore
    if reject is not None:
        reject_list = _get_reject_list(reject)
    else:
        reject_list = []

    # List of path to event.npy
    events_paths = _get_events_paths(epoch_dir, reject_list)
    stc_paths = _get_stc_paths(epoch_dir, area_name, reject_list)
    stc_paths, events_paths = _validate_paths(stc_paths, events_paths)

    if 0 < max_subjects < len(events_paths):
        events_paths = events_paths[:max_subjects]
        stc_paths = stc_paths[:max_subjects]
        logger.info(f"Using {max_subjects} subject data")

    # Generate x array
    if memmap:
        _generate_mmap(dst_dir, stc_paths, events_paths)
    else:
        _generate_data(dst_dir, stc_paths, events_paths)
    logger.info("Process terminated")

