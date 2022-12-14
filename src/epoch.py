import argparse
import logging
import os
import re
import sys
import traceback

from pathlib import Path
from typing import Callable, Tuple
import numpy as np
from mne import Epochs

from mne.io import Raw

from src.events.formatting import get_event_array, select_conditions
from src.utils.exceptions import SubjectNotProcessedError
from src.utils.file_access import get_mous_meg_channels, read_raw_format, load_json
from src.utils.logger import get_logger

logger = get_logger(file_name="epoch")
logger.setLevel(logging.INFO)


########################################################################################################################
# EPOCHING                                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #
# From the events data provided in the CSV segment the continuous data into epochs.                                    #
# Details at https://mne.tools/stable/auto_tutorials/epochs/index.html                                                 #
########################################################################################################################


def epoch(dst_dir: Path, events_dir: Path, subject: str,
          raw: Raw, events: np.array, mode,
          tmin: float, tmax: float, reject: dict, channel_reader: Callable,
          dictionary_path: str, simplify_mode: str, strict: bool, threshold: int) -> Epochs:
    """
    Epoch the subject data
    :param dst_dir: path to which the epochs object will be saved
    :param events_dir: path to events .csv files
    :param subject: name of the subject, e.g. sub-V1001
    :param raw: raw object
    :param events: events array (possibly downsampled)
    :param mode: whether to use `index` or `binary` noun vs. verb
    :param tmin: start time of the epoch
    :param tmax: end time of the epoch
    :param reject: rejection criteria
    :param channel_reader: a function for getting a list of relevant channels
    :param dictionary_path: path to .csv file containing POV information
    :param simplify_mode: how to select events. `sentence`, `list` or `both`
    :param strict: if true, validation of event will demand strict correspondence (max 1ms difference)
    :param threshold: maximum error in ms for validation. Only used if not `strict`
    """

    logger.info("Epoching the data...")

    # Get events data
    events = _read_events_file(events_dir, events, subject, mode, dictionary_path, simplify_mode, strict, threshold)

    # Get relevant channels
    picks = channel_reader(channels=raw.ch_names)

    epochs = None
    try:
        epochs = Epochs(raw, events, tmin=tmin, tmax=tmax,
                        picks=picks, preload=True, reject=reject, on_missing="warn")

    except ValueError as e:
        # Not all events are present for all subjects
        logger.exception(f"Missing event ids. Continuing {e}")

    # Save epochs to file
    _save_epochs(epochs, subject, dst_dir)

    # Save events to file
    fname = "events.npy"
    np.save(str(dst_dir / fname), epochs.events)

    return epochs


def _read_events_file(events_dir: Path, events: np.array, subject: str, mode,
                      dictionary_path: str, simplify_mode: str, strict: bool, threshold: int) \
        -> Tuple[np.array, dict]:
    """
    Convert the original events generated by mne.find_events and format + validate the events
    :param events_dir: directory containing .csv files about events
    :param events: tuple of (events, new_events) events array generated by MNE-python (plus events after downsampling)
    :param subject: subject name
    :param mode: whether to use `index` or `binary` noun vs. verb
    :param dictionary_path: path to .csv file containing POS information
    :param simplify_mode: how to select events. `sentence`, `list` or `both`
    :param strict: if true, validation of event will demand strict correspondence (max 1ms difference)
    :param threshold: maximum error in ms for validation. Only used if not `strict`
    :return:
        reformatted events array
    """
    logger.info(f"Reading events data for the subject {subject}")

    events_file = None

    # Find the corresponding event info file
    for file in os.listdir(events_dir):
        if re.match(fr"{subject}.*\.csv", file):  # otherwise finds "...-rejected.txt"
            events_file = file

    if events_file is None:
        msg = f"Events info for {subject} was not found"
        logger.exception(msg)
        raise SubjectNotProcessedError(FileNotFoundError, msg)

    # Validate events by comparing against .csv file
    event_path = events_dir / events_file
    events = get_event_array(events, event_path, Path(dictionary_path),
                             simplify_mode, strict=strict, threshold=threshold)

    # Filter conditions (sentence vs word list)
    events = select_conditions(events, mode=mode)

    return events


def _save_epochs(epochs: Epochs, subject: str, dst_dir: Path) -> None:
    """
    Save epochs to file
    :param epochs: epochs to be saved
    :param subject: subject name
    :param dst_dir: directory to save in
    """

    if epochs is not None:

        epoch_fname = f"{subject}-epo.fif"
        logger.debug(f"Writing {epoch_fname} epochs to file")

        try:
            epochs.save(str(dst_dir / epoch_fname), overwrite=True)
        except OSError as e:
            msg = f"Failed to write the file {dst_dir / epoch_fname}. {e}"
            logger.exception(msg)
            SubjectNotProcessedError(e, msg)


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Epoch raw files and save to a directory")

    # Read parameter from JSON
    parser.add_argument("--json_path", type=str, required=False, help="Path to JSON containing parameters")

    # Add arguments
    parser.add_argument("--dst_dir", type=str, required=False, help="Directory to save the results in")
    parser.add_argument("--raw_path", type=str, required=False, help="Path to raw file")
    parser.add_argument("--format", type=str, required=False, default="fif", help="Raw file format. Default is .fif")
    parser.add_argument("--events_dir", type=str, required=False, help="path to events .csv files")
    parser.add_argument("--subject", type=float, required=False, help="name of the subject, e.g. sub-V1001")
    parser.add_argument("--events_arr_dir", type=str, required=False, help="Path to directory containing events arrays")
    parser.add_argument("--mode", type=str, required=False, default="index",
                        help="whether to use `index` or `binary` noun vs. verb")
    parser.add_argument("--tmin", type=float, required=False, help="start time of the epoch")
    parser.add_argument("--tmax", type=float, required=False, help="end time of the epoch")
    parser.add_argument("--reject", type=float, required=False, help="rejection criteria for `mag` channels")
    parser.add_argument("--channel_reader", type=str, required=False, help="channel reader function name")
    parser.add_argument("--dictionary_path", type=str, required=False,
                        help="path to .csv file containing POV information")
    parser.add_argument("--simplify_mode", type=str, required=False, default="sentence",
                        help="how to select events. `sentence`, `list` or `both`. Default is `sentence`")
    parser.add_argument("--strict", type=bool, required=False, default=True,
                        help="If true, events sample value can only differ by 1")
    parser.add_argument("--threshold", type=int, required=False, help="Maximum acceptable error if not `strict` mode")

    # Convert to Path objects
    args = parser.parse_args()

    # Either from JSON or one by one
    if args.json_path:

        params = load_json(args.json_path)

        dst_dir, raw_path, file_format = params["dst-dir"], params["raw-path"], params["format"]
        events_dir, subject, events_arr_dir = params["events-dir"], params["subject"], params["events-arr-dir"]
        mode, tmin, tmax, reject = params["mode"], params["tmin"], params["tmax"], params["reject"]
        channel_reader, dictionary_path = params["channel-reader"], params["dictionary-path"]
        simplify_mode, strict, threshold = params["simplify-mode"], params["strict"], params["threshold"]
    else:
        dst_dir, raw_path, file_format, events_dir = args.dst_dir, args.raw_path, args.format, args.events_dir
        subject, events_arr_dir, mode, tmin, tmax = args.subject, args.events_arr_dir, args.mode, args.tmin, args.tmax
        reject, channel_reader, dictionary_path = args.reject, args.channel_reader, args.dictionary_path
        simplify_mode, strict, threshold = args.simplify_mode, args.strict, args.threshold

    # Convert to path objects
    raw_path = Path(raw_path)
    events_dir = Path(events_dir)
    events_arr_dir = Path(events_arr_dir)
    dst_dir = Path(dst_dir)
    dictionary_path = Path(dictionary_path)

    # Make sure directory exists
    if not dst_dir.exists():
        os.makedirs(dst_dir)

    # Convert to correct format
    reject = {"mag": reject}
    name_to_method = {"mous": get_mous_meg_channels}
    if channel_reader in name_to_method:
        channel_reader = name_to_method[channel_reader]
    else:
        raise ValueError(f"Unknown channel reader name {args.channel_reader}")

    return dst_dir, raw_path, file_format, events_dir, subject, events_arr_dir, mode, tmin, tmax, reject, \
           channel_reader, dictionary_path, simplify_mode, strict, threshold


if __name__ == "__main__":

    try:
        # Read parameter
        dst_dir, raw_path, file_format, events_dir, subject, events_arr_dir, mode, tmin, tmax, reject, channel_reader, \
            dictionary_path, simplify_mode, strict, threshold = get_args()

        # Read data
        raw = read_raw_format(path=raw_path, file_format=format)
        events = np.load(str(events_arr_dir / "events.npy"))
        new_events = np.load(str(events_arr_dir / "new-events.npy"))

        # Epoch
        epoch(dst_dir=dst_dir, events_dir=events_dir, subject=subject,
              raw=raw, events=(events, new_events), mode=mode,
              tmin=tmin, tmax=tmax, reject=reject, channel_reader=channel_reader,
              dictionary_path=str(dictionary_path), simplify_mode=simplify_mode, strict=strict, threshold=threshold)

    except FileNotFoundError as e:
        logger.exception(e.strerror)
        sys.exit(-1)

    except Exception as e:  # noqa

        logger.error(f"Unexpected exception during epoching. \n {traceback.format_exc()}")
        sys.exit(-1)

    logger.info("Epoching as finished.")
