import logging
import os
import pickle

from pathlib import Path
from datetime import datetime
from subprocess import call

from src.utils.logger import get_logger

logger = get_logger(file_name="slurm-tools")
logger.setLevel(logging.INFO)


# These are not valid subject numbers
NOT_SUBJECT = [14, 15, 18, 21, 23, 41, 43, 47, 51, 56, 60, 67, 82, 91, 112]

# Technical problems
PROB_SUBJECTS = []

MAX_ID = 117


# todo big comment at the top
# todo finish this file

########################################################################################################################
# SLURM UTILITY TOOLS                                                                                                  #
########################################################################################################################


def get_subject_list(n_max=MAX_ID):
    ignore_list = NOT_SUBJECT.extend(PROB_SUBJECTS)

    id_list, name_list = [], []
    for subject_id in range(MAX_ID):
        if subject_id in ignore_list:
            continue

        subject_name = f"sub-V1{str(subject_id).zfill(3)}"

        id_list.append(subject_id)
        name_list.append(subject_name)

        if len(id_list) > n_max:
            break

    return id_list, name_list


def get_area_id_list(parcellation="aparc"):
    return []


########################################################################################################################
# STATUS                                                                                                               #
########################################################################################################################


def init_status(name: str, n_tasks: int, mem: int, id_list: list):

    array = []
    for item_id in id_list:
        item_status = {"ID": item_id, "submitted": False, "submission-success": False, "completed": False,
                       "submission-time": None, "completion-time": None, "elapsed": None}
        array.append(item_status)

    return {"name": name, "n-tasks": n_tasks, "mem": mem, "completed": False, "array": array}


def update_status(status, success: bool):

    status["submitted"] = True
    status["submission-success"] = success
    status["submission-time"] = datetime.now()


def check_status(path: Path):

    if path.exists():
        with open(path, "rb") as handle:
            status = pickle.load(handle)
        return status
    else:
        return None


def save_status(status, path: Path):

    if not path.exists():
        os.makedirs(path)

    with open(path, "wb") as handle:
        pickle.dump(status, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_job_file(script: str, script_dir: Path, log_dir: Path, job_dir: Path,
                  name: str, array_str: str, output: str, params: dict,
                  n_tasks_per_node=1, mem_per_cpu=1):

    logger.info(f"Making a job file for the script {script}")

    # Make sure directories exist
    logs_dir = log_dir / name
    if not logs_dir.exists():
        os.makedirs(logs_dir)
    if not job_dir.exists():
        os.makedirs(job_dir)

    job_file = ""

    job_file += "#!/bin/bash\n"

    job_file += f"#SBATCH --job-name={name}\n"                      # name to be displayed for squeue etc.
    job_file += f"#SBATCH --array={array_str}\n"                    # job array, e.g. 1,2,3
    job_file += f"#SBATCH --ntasks-per-node={n_tasks_per_node}\n"   # number of tasks
    job_file += f"#SBATCH --chdir={logs_dir}\n"                     # working directory in logs_dir
    job_file += f"#SBATCH --mem-per-cpu={mem_per_cpu}gb\n"          # number of memory in GB
    job_file += f"#SBATCH --output={output}.out\n"                  # output file name

    job_file += f"scripts_dir={script_dir}\n"

    job_file += f"export PYTHONPATH=$PYTHONPATH:/data/home/hiroyoshi/scripts/meg-mvpa\n"
    job_file += f"python $scripts_dir/{script}"

    for name, value in params.items():

        if isinstance(value, list):
            job_file += f"--{name} {[str(item) + ' ' for item in value]}"
        else:
            job_file += f"--{name} {value}"
    job_file += "\n"

    job_path = job_dir / f"{name}.job"

    with open(job_path, "w") as f:
        f.write(job_file)

    logger.info(f"The job file was added to the job directory \n {job_file}")

    return job_path


def submit_subject_jobs(params: dict):

    ids, subjects = get_subject_list(params["n-max"])
    status = init_status(name=params["name"], n_tasks=params["n-tasks"], mem=params["mem"], id_list=subjects)

    results = {}
    for idx, (subject_id, subject) in enumerate(zip(ids, subjects)):

        # Check status is necessary
        subject_file = params["status-dir"] / params["name"] / f"{subject}-status.pickle"
        subject_status = check_status(subject_file)

        # Make job file
        if not subject_status:
            job_path = make_job_file(script=params["script"],
                                     script_dir=params["script-dir"], log_dir=params["log-dir"],
                                     job_dir=params["job-dir"], name=f"{params['name']}-{subject}",
                                     array_str=subject_id, output=f"{params['name']}-{subject}.out",
                                     params=params["params"],  # only python 3.6 < keeps original order
                                     n_tasks_per_node=params["slurm"]["n-tasks"], mem_per_cpu=params["slurm"]["mem"])

            # Submit job
            out = call(["sbatch", job_path])
            if out.returncode == 0:
                update_status(status["array"][idx], success=True)
                results["subject"] = "success"
            else:
                update_status(status["array"][idx], success=False)
                results["subject"] = "failure"

            # Save details
            save_status(status["array"][idx], path=subject_file)
    return results
