import json
import logging
import os
import pickle
import re

import mne
import numpy as np

from typing import Union, Callable, Tuple, List
from collections import OrderedDict
from pathlib import Path

from mne.io import read_raw_ctf, Raw
from mne import read_labels_from_annot, morph_labels

from src.utils.logger import get_logger

# todo: tidy this file

logger = get_logger(file_name="file_access")
logger.setLevel(logging.INFO)


def read_raw(src_dir: Path, dst_dir: Path, file_reader: Callable) -> Union[None, Raw]:
    """
    Reading the raw file for subjects from the path
    :param src_dir: directory containing all data
    :param dst_dir: directory in which results of processing will be stored
    :param file_reader: function specific to the particular data type
    :return: raw object for the subject
    """

    logger.debug(f"Reading raw file")

    raw = None

    try:
        raw = file_reader(src_dir)
    except FileNotFoundError as e:
        logger.exception(f"File {src_dir} was not found. Skipping this subject. {e.strerror}")

        # Remove dst_dir so as not to propagate errors
        try:
            if dst_dir.exists():
                os.remove(dst_dir)
        except OSError as e:
            logger.exception(f"Error: {e.strerror}")

    return raw


########################################################################################################################
# JSON file processes                                                                                                  #
########################################################################################################################


def load_json(json_path: Path):
    """
    For reading the JSON files from path
    :param: json_path: path to the JSON file
    """

    with open(json_path) as file:
        data = json.load(file)
    return data


def read_json(dir_path: Path, file_name: str):
    """
    A legacy method for reading JSON. Takes path to the parent directory and file
    :param dir_path: Path to the directory containing the JSON file
    :param file_name: name of the JSON file
    :return: content of the JSON file in dictionary format
    """

    json_dir = dir_path / file_name

    with open(json_dir) as file:
        data = json.load(file)

    return data


def write_json(dir_path: Path, file_name: str, data):
    """
    Write the python dictionary content to a JSON file
    :param dir_path: Path to the directory in which to save the file
    :param file_name: Name of the JSON file
    :param data: data to save
    :return: None
    """

    json_file = json.dumps(data, indent=4, ensure_ascii=False)

    with open(dir_path / file_name, "w") as file:
        file.write(json_file)


########################################################################################################################
# File processing specific to MOUS dataset                                                                             #
########################################################################################################################


def get_mous_raw_paths(data_dir: Path) -> List[Tuple[Path, str]]:
    """
    Find all paths to raw data in the MOUS dataset
    :param data_dir: path to directory containing all the data
    :return:
        list of tuples (path to data, name of the file)
    """

    logger.debug(f"Reading MOUS dataset paths from {data_dir}")

    path_list = []
    for file in os.listdir(data_dir):

        if re.match(r"^sub-[AV]\d+$", file):  # e.g. sub-V1001

            logger.debug(f"File {file} matched")

            subject_path = data_dir / file

            if os.path.isdir(subject_path):
                logger.debug(f"adding {subject_path} to the list")
                path_list.append((subject_path, file))  # Path and subject name
            else:
                msg = f"Expected a directory but got a file for the path {subject_path}."
                logger.exception(msg)
                raise TypeError(msg)

    return path_list


def read_mous_subject(subject_dir: Path, preload=True) -> Raw:
    """
    Read single subject raw file in MOUS dataset
    :param subject_dir: path to the directory of the subject
    :param preload:  if true, preload data into memory
    :return: raw object
    """

    logger.debug(f"Parsing {subject_dir}")

    meg_dir = subject_dir / "meg"
    ctf_dir = None

    for file_name in os.listdir(meg_dir):
        logger.debug(f"Parsing {meg_dir} {file_name}")

        if re.match(r"^sub-[AV]\d+_task-visual_meg\.ds$", file_name):
            ctf_dir = meg_dir / file_name

    if ctf_dir is None:
        msg = f"The CTF directory in {meg_dir} was not found."
        logger.exception(msg)
        raise FileNotFoundError(msg)

    raw = read_raw_ctf(str(ctf_dir), preload=preload)

    logger.debug(f"File {ctf_dir} successfully read")
    return raw


def get_mous_meg_channels(channels: list) -> List[str]:
    """
    Get list of MEG channel names in MOUS dataset
    :param channels: List of channels to include (string)
    :return:
        picks: list of names of MEG channels
    """

    picks = []
    for channel in channels:
        if re.match(r"M.*4304", channel):  # Select MEG gradiometers
            picks.append(channel)
    return picks


def read_raw_format(path, file_format):
    """
    Read MEG data in a specific data format (FIF or CTF)
    :param path: path to the data
    :param file_format: name of the format (`fif` or `ctf`)
    :return: Raw object
    """

    if file_format == "fif":
        raw = mne.io.read_raw(path)
        return raw
    elif file_format == "ctf":
        raw = read_raw_ctf(path)
        return raw


def _get_file_names(results_dir: Path):
    """
    Get a list of file names of pickle files containing decoding score results
    :param results_dir: path to directory containing results
    :return:
        Ordered dictionary: {name of the file: path}
    """

    name_to_file = {}
    for file in os.listdir(results_dir):
        if re.match(r"^\..*", file):  # skip hidden files, e.g. .DS_Store...
            continue

        name = re.sub(r"\.pickle", "", file)
        name_to_file[name] = file

    name_to_file = OrderedDict(sorted(name_to_file.items()))  # todo warning wrong type
    return name_to_file


def read_dict(path: Path) -> dict:
    """
    Read pickle object containing a python dictionary
    :param path: path to the file
    :return:
        dictionary
    """
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return data


def read_labels(parc, subject, hemi, subjects_dir):
    """
    todo comment
    :param parc: todo parc
    :param subject: todo subject
    :param hemi: todo hemi
    :param subjects_dir: todo subjects_dir
    :return: todo return
    """

    labels = read_labels_from_annot("fsaverage", parc=parc, hemi=hemi, subjects_dir=subjects_dir)

    if subject != "fsaverage":
        labels = morph_labels(labels, subject_to=subject, subject_from="fsaverage",
                              subjects_dir=subjects_dir, surf_name="white")
    return labels


def read_scores(results_dir: Path):
    """
    todo comment
    :param results_dir: todo results_dir
    :return: todo return
    """

    name_to_file = _get_file_names(results_dir)  # ordered dict to load in correct order

    meta, data = None, []
    for name, file in name_to_file.items():

        with open(results_dir / file, "rb") as handle:
            results = pickle.load(handle)

        if not meta:  # metadata is the same for all
            mata = results["meta"]

        data.append(results["data"]["scores"])

    data = np.stack(data, axis=2)
    return meta, data


def read_data(data_dir: Path):
    """
    todo comment
    :param data_dir: todo data_dir
    :return: todo return
    """

    json_path = data_dir / "x_shape.json"
    if json_path.exists():

        x_path = data_dir / "x.dat"

        if x_path.exists():
            json_data = load_json(data_dir / "x_shape.json")
            print(tuple(json_data["shape"]))
            x = np.memmap(str(x_path), dtype="float64", mode="r+", shape=tuple(json_data["shape"]))
            return x
        else:
            raise FileNotFoundError(f"'x.dat' file was not found in {data_dir}")
    else:
        x_path = data_dir / "x.npy"

        if x_path.exists():
            x = np.load(str(x_path))
            return x
        else:
            raise FileNotFoundError(f"Neither JSON file nor 'x.npy' file was found in {data_dir}")


def convert_bti_to_fif(meg_dir: Path, dst: Path) -> None:
    """
    Convert BTI format into FIF format
    :param meg_dir: path to MEG directory containing main file, config file and header
    :param dst: path to save data
    """

    # Get raw MEG data

    raw = mne.io.read_raw_bti(pdf_fname=str(meg_dir / "c,rfDC"),
                              config_fname=str(meg_dir / "config"),
                              head_shape_fname=str(meg_dir / "hs_file"),
                              preload=True)

    # Drop incorrect EEG lines
    raw.drop_channels(["VEOG", "HEOG", "OO", "EEG 001"])

    # Save
    raw.save(str(dst))
