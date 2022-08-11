from pathlib import Path
from mne.coreg import Coregistration
from mne import Info, setup_source_space, make_bem_model, make_bem_solution, make_forward_solution, Transform, Forward

########################################################################################################################
# COREGISTRATION                                                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #
# Handles coregistration calculation for MEG data. Subject specific forward models are generated from coregistraion.   #
# Details about coregistration is found here: https://mne.tools/stable/auto_tutorials/forward/25_automated_coreg.html  #
########################################################################################################################


def get_trans(subject: str, subjects_dir: Path, info: Info) -> Transform:
    """
    Get Transform instance from the movement measurements.
    https://mne.tools/stable/auto_tutorials/forward/25_automated_coreg.html
    :param subject: name of the subject
    :param subjects_dir: freesurfer subject directory
    :param info: measurement info to be fed to the algorithm
    :return: Transform data
    """

    coreg = Coregistration(info, subject=subject, subjects_dir=subjects_dir, fiducials="auto")
    coreg.fit_icp(n_iterations=6, nasion_weight=2.)
    coreg.omit_head_shape_points(distance=5. / 1000)
    coreg.fit_icp(n_iterations=20, nasion_weight=10.)
    return coreg.trans


def get_forward(info: Info, trans: str, subject: str, subjects_dir: Path, layers: int) -> Forward:
    """
    Get forward model specific for the subject. https://mne.tools/stable/auto_tutorials/forward/30_forward.html
    :param info: Info object about the data
    :param trans: Transform for the measurement
    :param subject: subject name
    :param subjects_dir: path to freesurfer directory
    :param layers: whether to use 1 or 3 layers
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
