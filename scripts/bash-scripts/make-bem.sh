#!/bin/bash

# Script for performing FreeSurfer reconstruction of subject’s brain from nifti file
# Executing this will create SUBJECT/NAME with several folders (bem, label, mri etc.)
# See https://mne.tools/stable/auto_tutorials/forward/10_background_freesurfer.html

# Edit these parameters
HOME=/path/to/freesurfer          # path to freesurfer home
NAME="sub-V1001"                  # name of the subject, e.g. sub-V1001

export FREESURFER_HOME=$HOME
source $FREESURFER_HOME/SetUpFreeSurfer.sh

mne watershed_bem -s $NAME --overwrite
