#!/bin/bash

# Script for performing FreeSurfer reconstruction of subject’s brain from nifti file
# Executing this will create SUBJECT/NAME with several folders (bem, label, mri etc.)
# See https://mne.tools/stable/auto_tutorials/forward/10_background_freesurfer.html

# Edit these parameters
HOME=/path/to/freesurfer          # path to freesurfer home
SUBJECTS=/path/to/subjects        # path to subjects_dir
NAME="sub-V1001"                  # name of the subject, e.g. sub-V1001
NIFTI=/path/to/nifti              # path to Nifti file, e.g. .../sub-V1001/anat/sub-V1001_T1w.nii

export FREESURFER_HOME=$HOME
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=$SUBJECTS

recon-all -i $NIFTI -s NAME -all
