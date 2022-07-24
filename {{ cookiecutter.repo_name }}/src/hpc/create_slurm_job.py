import logging
import os
import re
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.option(
    "--code",
    default="dvc",
    type=str,
    help="Specify custom code to run or indicate 'dvc' or 'jupyter' for for dvc repro or jupyter lab, respectively",
)
@click.option(
    "--conda-env",
    default="marvl_pytorch",
    type=str,
    help="Conda environment to activate before running code",
)
@click.option("--cpu", "-c", default=2, type=int, help="Number of cores per task")
@click.option("--email", "-e", default=None, help="Email address for notifications")
@click.option("--gpu", "-g", default=None, type=int, help="Number of GPUs requested")
@click.option(
    "--jupy_ip", default="0.0.0.0", type=str, help="IP address for jupyter lab hosting"
)
@click.option(
    "--job_name", "-j", default="bash", type=str, help="Name for job submission"
)
@click.option(
    "--mail-type",
    default="END,FAIL",
    type=str,
    help="Events for email notification (NONE, BEGIN, END, FAIL, ALL)",
)
@click.option("--mem", "-m", default=8, type=int, help="RAM per processor")
@click.option(
    "--nodelist",
    "-nl",
    default="pasteur1"
    if re.search("sail", os.environ["HOME"], re.IGNORECASE)
    else None,
)
@click.option(
    "--nodes", "-n", default=1, type=int, help="Maximum number of nodes to be allocated"
)
@click.option(
    "--partition",
    "-p",
    default="pasteur"
    if re.search("sail", os.environ["HOME"], re.IGNORECASE)
    else "syyeung",
    type=str,
    help="Partition to use",
)
@click.option("--port", default=8800, type=int, help="Port for jupyter lab hosting")
@click.option(
    "--qos",
    "-q",
    default="normal",
    type=str,
    help="Slurm quality of service specification",
)
@click.option(
    "--time", "-t", default="2:00:00", type=str, help="Wall time limit days-HH:mm:ss"
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Print verbose output.",
)
def main(
    code: str = "dvc",
    conda_env: str = "marvl_pytorch",
    cpu: int = 2,
    email=None,
    gpu=None,
    job_name: str = "bash",
    job_submission_file: str = "slurm_job.sh",
    jupy_ip: str = "0.0.0.0",
    mail_type: str = "END,FAIL",
    mem: int = 16,
    nodelist=None,
    nodes: int = 1,
    partition: str = "pasteur",
    port: int = 8800,
    qos: str = "normal",
    time: str = "2:00:00",
    verbose: bool = False,
) -> int:
    """Creates shell script for slurm job submission

    Output: Creates shell script file for running slurm jobs
    """

    # set dirs
    script_dir = Path(__file__).resolve().parents[0]  # project_dir/src/hpc
    project_dir = script_dir.parents[2]  # project_dir/
    conda_profile = Path.home().joinpath("miniconda3", "etc", "profile.d", "conda.sh")

    # validate inputs
    validate_inputs(email, mail_type, partition, port, time)

    job_submission_file = script_dir.joinpath(job_submission_file).resolve()
    if verbose:
        logger = logging.getLogger(__name__)
        logger.info(
            f"Creating slurm job submission file:\n"
            f"\tProject directory:\t{project_dir}\n"
            f"\tJob_submission_file:\t{job_submission_file}\n"
            f"\t#################### SETTINGS ####################\n"
            f"\t#SBATCH --job-name='{job_name}'\n"
            f"\t#SBATCH --partition='{partition}' --qos={qos}\n"
            f"\t#SBATCH --nodelist='{nodelist}'\n"
            f"\t#SBATCH --output=./logs/{job_name}_%j.out\n"
            f"\t#SBATCH --error=./logs/{job_name}_%j.err\n"
            f"\t#SBATCH --time={time}\n"
            f"\t#SBATCH --nodes={nodes}\n"
            f"\t#SBATCH --mem={mem}gb\n"
            f"\t#SBATCH --cpus-per-task={cpu}\n"
            f"\t#SBATCH --gres=gpu:'{gpu}'\n"
            f"\t#SBATCH --mail-user='{email}'\n"
            f"\t#SBATCH --mail-type='{mail_type.upper() if email else None}'\n"
        )
        logger.info(f"Setting up slurm job submission to use conda env {conda_env}")

    with open(job_submission_file, "w") as fh:
        fh.writelines("#!/bin/bash\n# -*- coding: utf-8 -*-\nset -o pipefail\n\n")
        fh.writelines(
            f"#SBATCH --job-name='{job_name}'\n"
            f"#SBATCH --partition='{partition}' --qos={qos}\n"
        )
        if partition == "pasteur" and nodelist:
            fh.writelines(f"#SBATCH --nodelist='{nodelist}'\n")

        fh.writelines(
            f"#SBATCH --output=./logs/{job_name}_%j.out # STDOUT logs %j=job id\n"
            f"#SBATCH --error=./logs/{job_name}_%j.err # Error logs %j=job id \n"
            f"#SBATCH --time={time} # Wall time limit days-HH:mm:ss\n"
            f"#SBATCH --nodes={nodes} # Maximum number of nodes to be allocated\n"
            f"#SBATCH --mem={mem}gb # Memory (e.g., RAM) in gb \n"
            f"#SBATCH --cpus-per-task={cpu} # Number of cores per task\n"
        )

        if gpu is not None:
            fh.writelines(f"#SBATCH --gres=gpu:'{gpu}'\n")
        if email is not None:
            fh.writelines(f"#SBATCH --mail-user='{email}'\n")
            fh.writelines(f"#SBATCH --mail-type='{mail_type.upper()}'\n")

        if verbose:
            fh.writelines(
                "\n# print information\n"
                'echo -e "SLURM_JOBID:\\t$SLURM_JOBID\n'
                "SLURM_JOB_NODELIST:\\t$SLURM_JOB_NODELIST\n"
                "SLURM_NNODES:\\t$SLURM_NNODES\n"
                "SLURMTMPDIR:\\t$SLURMTMPDIR\n"
                "Working directory:\\t$SLURM_SUBMIT_DIR\n\n"
                'Allocating resources..."\n\n'
            )
        if verbose:
            fh.writelines(
                "# sample process (list hostnames of the nodes you've requested)\n"
                "NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l` \n"
                'echo -e "NPROCS:\\t$NPROCS"\n\n'
            )

            fh.writelines(
                "# list the allocated gpu, if desired\n"
                "#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery\n"
            )

        fh.writelines(
            '\necho -e "Resources are allocated."\n'
            f"if [[ -f {conda_profile} ]]; then\n\t"
            f'echo -e "Activating conda environment {conda_env}."\n\t'
            f"source {conda_profile}\n\tconda activate {conda_env}\n"
            f"else\n\t"
            f"echo -e 'FileNotFoundError: {conda_profile}'\nexit 1\n"
            "fi\n\n"
        )

        if code.lower() == "dvc":
            if verbose:
                logger.info(f"Setting up slurm job submission to run DVC")

            fh.writelines(
                '\n ## run dvc\necho -e "Running DVC repro..."\n'
                f'cd "{Path(project_dir).resolve()}"\n'
                "dvc pull -r origin\n"
                "dvc repro\n\n"
                "# Create report comparing metrics to master\n"
                "git fetch --prune\n"
                f"echo -e '# Report\\n## Parameters\\n' >> {project_dir.joinpath('reports', 'results.md')}\n"
                f"dvc params diff master --show-md >> {project_dir.joinpath('reports', 'results.md')}\n"
                f"dvc metrics diff --show-md master >> {project_dir.joinpath('reports', 'results.md')}\n"
                "\n"
                'echo -e "Exit code:\t$?"'
            )
        elif code.lower() == "jupyter":
            if verbose:
                logger.info(
                    f"Setting up slurm job submission to host Jupyter lab "
                    f"listening on port:{port}, ip:{jupy_ip}"
                )

            fh.writelines(
                '\n## run jupyter\necho -e "Running Jupyter Lab...\\n'
                f'\\tListening on port:{port} ip:{jupy_ip}"\n\n'
                f'cd "{Path(project_dir).resolve()}"\n'
                f"jupyter lab --port={port} --ip={jupy_ip}\n\n"
                'echo -e "Exit code:\t$?"'
            )
        else:
            if verbose:
                logger.info(f"Setting up slurm job submission to run: " f"{code}")
                fh.writelines(
                    'echo -e "Running code..."\n'
                    f'cd "{Path(project_dir).resolve()}"\n'
                    f"{code}\n\n"
                    'echo -e "Exit code:\t$?"'
                )

    if verbose:
        logger.info(
            f"Completed slurm job file creation:\t"
            f"{job_submission_file.relative_to(os.getcwd())}"
        )
        # os.system(f"sbatch {job_submission_file}")


def validate_inputs(email, mail_type, partition, port, time):
    # error checking
    valid_mail_types = ["none", "begin", "end", "fail", "all"]
    for elem in mail_type.lower().split(","):
        assert elem in valid_mail_types, ValueError(
            f"Expected --mail_type to be in {valid_mail_types} but received {elem}"
        )

    valid_partition = [
        "syyeung",
        "pasteur",
    ]
    assert partition in valid_partition, ValueError(
        f"Expected --partition to be in {valid_partition} but received {partition}"
    )

    re_time_format = re.compile("\d?-?\d{1,2}:\d{2}:\d{2}")
    if re_time_format.match(time) is None:
        raise ValueError(
            f"Expected --time in format days-HH:mm:ss, but received '{time}'"
        )

    re_email_format = re.compile("\w.*@\w*\.(edu|com|org|net)")
    if email and re_email_format.match(email) is None:
        raise ValueError(
            f"Expected --email in format '\w.*@\w*\.(edu|com|org|net)', but received '{email}'"
        )

    assert port > 0 and port < 2 ** 16, ValueError(
        f"Expected port in range 1:{2 ** 16}, received {port}"
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
