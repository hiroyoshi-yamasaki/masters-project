import argparse
import logging
import os
import sys
import traceback

from pathlib import Path
from typing import Tuple

import numpy as np
from mne import find_events
from mne.io import Raw

from src.utils.exceptions import SubjectNotProcessedError
from src.utils.file_access import read_raw_format, load_json
from src.utils.logger import get_logger

logger = get_logger(file_name="downsample")
logger.setLevel(logging.INFO)


########################################################################################################################
# DOWNSAMPLING                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #
# Downsample to speedup the processes.                                                                                 #
# Details in: https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html                       #
########################################################################################################################


def downsample(raw: Raw, sfreq: int, n_jobs, mous=True) -> Tuple[Raw, np.array, np.array]:
    """
    Downsample to some lower sampling frequency
    :param raw: raw object
    :param sfreq: sampling frequency
    :param n_jobs: number of jobs for parallelism
    :param mous: if true, use 'UPPT001' and 'UPPT002' as channels
    :return:
        raw: resampled raw object
        events: original events (needed for validating events)
        new_events: events with new sampling frequency
    """

    logger.info(f"Downsampling to {sfreq} Hz")

    # Find events (needed whether it is downsampled or not)
    try:
        if mous:
            events = find_events(raw, stim_channel=["UPPT001", "UPPT002"], min_duration=2 / raw.info["sfreq"])
        else:
            events = find_events(raw)
    except ValueError as e:
        logger.exception(f"Issue with shortest event. Needs manual inspection {e}")
        raise SubjectNotProcessedError(e)

    # If sampling frequency is specified, downsample
    if sfreq > 0 and not None:
        logger.debug(f"Resampling at {sfreq} Hz")

        raw, new_events = raw.resample(sfreq=sfreq, events=events, n_jobs=n_jobs)

        return raw, events, new_events
    else:
        return raw, events, events


def get_args():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Downsample raw file and save the results")

    # Read parameter from JSON
    parser.add_argument("--json_path", type=str, required=False, help="Path to JSON containing parameters")

    # Read individual arguments
    parser.add_argument("--raw_path", type=str, required=False, help="Path to raw file")
    parser.add_argument("--format", type=str, required=False, default="fif", help="Raw file format. Default is .fif")
    parser.add_argument("--sfreq", type=float, required=False, help="Sampling frequency")
    parser.add_argument("--n_jobs", type=int, required=False, default=1, help="Number of jobs to use. Default is 1.")
    parser.add_argument("--dst_dir", type=str, required=False, help="Directory to save the results in")
    parser.add_argument("--name", type=str, required=False, default="", help="File name. Default is empty.")
    parser.add_argument("--mous", type=bool, required=False, default=True, help="If true, use UPPT001/002")

    args = parser.parse_args()

    # Either from JSON or one by one
    if args.json_path:
        params = load_json(args.json_path)
        raw_path, format, sfreq = params["raw-path"], params["format"], params["sfreq"]
        n_jobs, dst_dir, name, mous = params["n-jobs"], params["dst-dir"], params["name"], params["mous"]
    else:
        raw_path, format, sfreq, n_jobs, dst_dir, name, mous = \
            args.raw_path, args.format, args.sfreq, args.n_jobs, args.dst_dir, args.name, args.mous

    # Convert to path object
    raw_path = Path(raw_path)
    dst_dir = Path(dst_dir)

    # Make sure the directory exists
    if not dst_dir.exists():
        os.makedirs(dst_dir)

    return raw_path, format, sfreq, n_jobs, dst_dir, name, mous


if __name__ == "__main__":

    try:
        # Read parameters
        raw_path, format, sfreq, n_jobs, dst_dir, name, mous = get_args()

        # Read raw
        raw = read_raw_format(raw_path, format)

        # Downsample
        raw, events, new_events = downsample(raw=raw, sfreq=sfreq, n_jobs=n_jobs, mous=mous)

        # Save to file
        if not dst_dir.exists():
            os.makedirs(dst_dir)
        raw.save(str(dst_dir / f"{name}-downsampled-{sfreq}Hz-raw.fif"), overwrite=True)
        np.save(str(dst_dir / f"{name}-downsampled-{sfreq}Hz-original-events.npy"), events)
        np.save(str(dst_dir / f"{name}-downsampled-{sfreq}Hz-new-events.npy"), new_events)

    except FileNotFoundError as e:
        logger.exception(e.strerror)
        sys.exit(-1)

    except Exception as e:  # noqa

        logger.error(f"Unexpected exception during filtering. \n {traceback.format_exc()}")
        sys.exit(-1)

    logger.info(f"Downsampling finished.")
