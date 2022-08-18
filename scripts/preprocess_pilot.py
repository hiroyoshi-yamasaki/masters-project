import argparse
from pathlib import Path
import mne


def get_raw(fname: Path, config: Path, head: Path, eeg_header: Path, l_freq: float, h_freq: float, dst_dir: Path):

    # Get raw MEG data
    raw = mne.io.read_raw_bti(pdf_fname=str(fname), config_fname=str(config), head_shape_fname=str(head), preload=True)

    # Drop incorrect EEG lines
    raw.drop_channels(["VEOG", "HEOG", "OO", "EEG 001"])

    # Bandpass filter
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)

    # Get raw EEG data
    eeg_raw = mne.io.read_raw_brainvision(eeg_header, preload=True)

    eeg_raw = eeg_raw.resample(sfreq=raw.info["sfreq"])     # must have the same sfreq as MEG
    eeg_raw = eeg_raw.filter(l_freq=l_freq, h_freq=h_freq)  # must have the same filtering as MEG

    # Crop (by aligning to events)
    events = mne.events_from_annotations(eeg_raw)
    first_sample = events[0][1, 0]  # events is a list with metadata (first element is the array)
    eeg_raw.crop(eeg_raw.times[first_sample], eeg_raw.times[first_sample] + eeg_raw.times[raw.n_times - 1])

    # Add EEG to MEG
    raw.add_channels([eeg_raw], force_update_info=True)

    raw.save(str(dst_dir / "raw.fif"))


def get_args():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert to FIF format")

    # Add arguments
    parser.add_argument("--meg_dir", type=str, required=True, help="path to MEG directory")
    parser.add_argument("--eeg_header", type=str, required=True, help="path to EEG header")
    parser.add_argument("--h_freq", type=float, required=True, help="h_freq")
    parser.add_argument("--l_freq", type=float, required=True, help="l_freq")
    parser.add_argument("--dst_dir", type=str, required=True, help="path to save the FIF file")

    # Get argument
    args = parser.parse_args()

    return Path(args.meg_dir), Path(args.eeg_header), args.h_freq, args.l_freq, Path(args.dst_dir)


if __name__ == "__main__":

    meg_dir, eeg_header, h_freq, l_freq, dst_dir = get_args()
    fname = meg_dir / "data"
    config = meg_dir / "config"
    head = meg_dir / "hs_file"

    get_raw(fname, config, head, eeg_header, l_freq, h_freq, dst_dir)
