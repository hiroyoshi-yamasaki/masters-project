import argparse
import logging
import os
import sys
import traceback

from pathlib import Path

from mne.io import Raw

from src.utils.file_access import read_raw_format, load_json
from src.utils.logger import get_logger

logger = get_logger(file_name="filter")
logger.setLevel(logging.INFO)


########################################################################################################################
# FILTERING                                                                                                            #
########################################################################################################################
# Filter the data by frequency to remove noise                                                                         #
# Details in: https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html                       #
########################################################################################################################


def apply_filter(raw: Raw, l_freq: int, h_freq: int, notch: list, n_jobs=1) -> Raw:
    """
    Apply band-pass filter and notch filter
    :param raw: raw file to apply filter to
    :param l_freq: lower frequency limit
    :param h_freq: upper frequency limit
    :param notch: list frequencies
    :param n_jobs: number of jobs for parallelism
    :return: filtered raw
    """

    logger.info(f"Filtering at high pass {l_freq} Hz, low pass {h_freq} and notches {notch}. n_jobs = {n_jobs}")

    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)

    if len(notch) > 0:
        raw = raw.notch_filter(freqs=notch)

    return raw


def get_args():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Filter raw file and save the results")

    # Read parameter from JSON
    parser.add_argument("--json_path", type=str, required=False, help="Path to JSON containing parameters")

    # Add arguments
    parser.add_argument("--raw_path", type=str, required=False, help="Path to raw file")
    parser.add_argument("--format", type=str, required=False, default="fif", help="Raw file format. Default is .fif")
    parser.add_argument("--l_freq", type=float, required=False, help="High pass frequency")
    parser.add_argument("--h_freq", type=float, required=False, help="Low pass frequency")
    parser.add_argument("--notch", nargs="+", type=float, required=False, help="List of notch filter frequencies")
    parser.add_argument("--dst_dir", type=str, required=False, help="Directory to save the results in")
    parser.add_argument("--name", type=str, required=False, default="", help="File name. Default is empty.")
    parser.add_argument("--n_jobs", type=int, required=False, default=1, help="Number of jobs to use. Default is 1")

    args = parser.parse_args()

    # Either from JSON or one by one
    if args.json_path:

        params = load_json(args.json_path)

        raw_path, format, l_freq, h_freq = params["raw-path"], params["format"], params["l-freq"], params["h-freq"]
        notch, dst_dir, name, n_jobs = params["notch"], params["dst-dir"], params["name"], params["n-jobs"]
    else:
        raw_path, format, l_freq, h_freq, notch, dst_dir, name, n_jobs = \
            args.raw_path, args.format, args.l_freq, args.h_freq, args.notch, args.dst_dir, args.name, args.n_jobs

    # Convert to path objects
    raw_path = Path(raw_path)
    dst_dir = Path(dst_dir)

    # Make sure the directory exists
    if not dst_dir.exists():
        os.makedirs(dst_dir)

    return raw_path, format, l_freq, h_freq, notch, dst_dir, name, n_jobs


if __name__ == "__main__":

    try:
        # Read parameters
        raw_path, format, l_freq, h_freq, notch, dst_dir, name, n_jobs = get_args()

        # Read raw
        raw = read_raw_format(path=raw_path, format=format).load_data()

        # Filter
        raw = apply_filter(raw=raw, l_freq=l_freq, h_freq=h_freq, notch=notch, n_jobs=n_jobs)

        # Save to file
        raw.save(str(dst_dir / f"{name}-filtered-raw.fif"), overwrite=True)

    except FileNotFoundError as e:
        logger.exception(e.strerror)
        sys.exit(-1)

    except Exception as e:  # noqa

        logger.error(f"Unexpected exception during filtering. \n {traceback.format_exc()}")
        sys.exit(-1)

    logger.info(f"Filtering finished.")
