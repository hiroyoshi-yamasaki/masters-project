import argparse
import logging
import os
import sys
import traceback

from pathlib import Path

from mne.io import Raw
from mne.preprocessing import ICA

from src.utils.file_access import read_raw_format, load_json
from src.utils.logger import get_logger

logger = get_logger(file_name="artifact")
logger.setLevel(logging.INFO)


########################################################################################################################
# ARTIFACT REMOVAL                                                                                                     #
########################################################################################################################


def remove_artifacts(raw: Raw, n_components: float, eog_channels=None, ecg_channel=None,
                     save_ica=True, apply=False, dst_dir=None, n_jobs=1) -> Raw:
    """
    Perform artifact removal using ICA
    :param raw: mne raw object
    :param n_components: number of components to use for ICA
    :param eog_channels: list of channel names to be used as EOG channels
    :param ecg_channel: the name of the channel to be used as the ECG channel
    :param save_ica: if true, save the ICA object
    :param apply: if true, apply ICA reconstruction to the raw
    :param dst_dir: path to directory to save the results in
    :param n_jobs: number of jobs for parallelism
    :return: raw: repaired raw
    """

    logger.info("Removing artifacts")

    if eog_channels is None and ecg_channel is None:
        logger.debug("Skipping artifact repair")
        return raw

    # Perform ICA
    logger.info(f"Starting ICA with {n_components} components")

    # Needs to be high-pass filtered first
    filtered_raw = raw.copy().filter(l_freq=1., h_freq=None, n_jobs=n_jobs)

    ica = ICA(n_components=n_components)
    ica.fit(filtered_raw)

    ica.exclude = []

    if apply:
        # Remove ocular artifacts
        if eog_channels is not None:
            logger.debug("Repairing ocular artifacts")
            eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_channels, verbose=True)
            ica.exclude = eog_indices

        # Remove heartbeat artifacts
        if ecg_channel is not None:
            logger.debug("Repairing heartbeat artifacts")
            ecg_indices, _ = ica.find_bads_eog(raw, ch_name=ecg_channel, verbose=True)
            ica.exclude = ecg_indices

        logger.info(f"Total of {len(ica.exclude)} components removed")

        ica.apply(raw)
        raw.save(dst_dir / f"ica-raw.fif", overwrite=True)

    if save_ica:
        ica.save(dst_dir / f"ica.fif", overwrite=True)
    return raw


def get_args():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Perform ICA artifact removal")

    # Read parameter from JSON
    parser.add_argument("--json_path", type=str, required=False, help="Path to JSON containing parameters")

    # Add arguments
    parser.add_argument("--raw_path", type=str, required=False, help="Path to raw file")
    parser.add_argument("--dst_dir", type=str, required=False, help="Directory to save the results in")
    parser.add_argument("--format", type=str, required=False, default="fif", help="Raw file format. Default is .fif")
    parser.add_argument("--n_components", type=float, required=False, default=0.99,
                        help="Number of components for ICA. Default is 0.99 explained variance (see MNE documentation)")
    parser.add_argument("--eog_channels", type=str, required=False, help="Names of the EOG channels")
    parser.add_argument("--ecg_channel", type=str, required=False, help="Name of the ECG channel")
    parser.add_argument("--save_ica", type=bool, required=False, default=True,
                        help="If true, save ICA object. Default is True")
    parser.add_argument("--apply", type=bool, required=False, default=True,
                        help="If true, apply ICA reconstruction to the raw")
    parser.add_argument("--n_jobs", type=int, required=False, default=1, help="Number of jobs to use. Default is 1")

    args = parser.parse_args()

    # Either from JSON or one by one
    if args.json_path:

        params = load_json(args.json_path)

        raw_path, file_format, n_components = params["raw-path"], params["format"], params["n-components"]
        eog_channels, ecg_channel = params["eog-channels"], params["ecg-channel"]
        save_ica, apply, n_jobs, dst_dir = params["save-ica"], params["apply"], params["n-jobs"], params["dst-dir"]
    else:
        raw_path, file_format, n_components = args.raw_path, args.format, args.n_components
        eog_channels, ecg_channel = args.eog_channels, args.ecg_channel
        save_ica, apply, n_jobs, dst_dir = args.save_ica, args.apply, args.n_jobs, args.dst_dir

    # Convert to path object
    raw_path = Path(raw_path)
    dst_dir = Path(dst_dir)

    # Make sure the directory exists
    if not dst_dir.exists():
        os.makedirs(dst_dir)

    return raw_path, file_format, n_components, eog_channels, ecg_channel, save_ica, apply, n_jobs, dst_dir


if __name__ == "__main__":

    try:
        # Read parameters
        raw_path, file_format, n_components, eog_channels, ecg_channel, save_ica, apply, n_jobs, dst_dir = get_args()

        # Read raw
        raw = read_raw_format(path=raw_path, file_format=file_format).load_data()

        # ICA
        raw = remove_artifacts(raw=raw, n_components=n_components,
                               eog_channels=eog_channels, ecg_channel=ecg_channel,
                               save_ica=save_ica, apply=apply, n_jobs=n_jobs, dst_dir=dst_dir)

    except FileNotFoundError as e:
        logger.exception(e.strerror)
        sys.exit(-1)

    except Exception as e:  # noqa

        logger.error(f"Unexpected exception during filtering. \n {traceback.format_exc()}")
        sys.exit(-1)

    logger.info(f"ICA artifact removal finished.")
