#!/usr/bin/env python3

import logging
import os
import re
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument(
    "input_dir",
    default=Path("../").resolve(),
    type=click.Path(exists=True),
)
@click.option("--code", default="dvc", type=str)
@click.option("--conda_env", default="marvl_pytorch", type=str)
@click.option("--cpu", "-c", default=2, type=int)
@click.option("--email", "-e", default=None)
@click.option("--gpu", "-g", default=None)
@click.option("--jupy_ip", default="0.0.0.0", type=str)
@click.option("--job_name", "-j", default="bash", type=str)
@click.option("--mail-type", default="END,FAIL")
@click.option("--mem", "-m", default=8, type=int)
@click.option("--nodelist", "-nl", default="pasteur1", type=str)
@click.option("--nodes", "-n", default=1, type=int)
@click.option("--partition", "-p", default="pasteur", type=str)
@click.option("--port", default=8800, type=int)
@click.option("--qos", "-q", default="normal", type=str)
@click.option("--time", "-t", default="2:00:00", type=str)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Switch to toggle verbose output.",
)
def main(
    input_dir: str,
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
    nodelist: str = "pasteur1",
    nodes: int = 1,
    partition: str = "pasteur",
    port: int = 8800,
    qos: str = "normal",
    time: str = "2:00:00",
    verbose: bool = False,
) -> int:
    """Creates shell script for slurm job submission

    Args:
        code: specify code to run
        conda_env: conda environment to activate before running code
        cpu: Number of cores per task
        email: Email address for notifications
        gpu: GPUs requested
        job_name: str = "bash",
        mail_type: Mail events (NONE, BEGIN, END, FAIL, ALL)
        mem: Memory (e.g., RAM) per processor
        nodelist: Nodelist to use "pasteur1",
        nodes: Maximum number of nodes to be allocated
        partition: Partition to use
        qos: str = "normal",
        time: str = "2:00:00",
        verbose: bool = False,

    Output: Creates shell script file for running slurm jobs
    """

    validate_inputs(email, mail_type, partition, port, time)

    job_submission_file = Path("./src/hpc").joinpath(job_submission_file).resolve()
    if verbose:
        logger = logging.getLogger(__name__)
        logger.info(
            f"Creating slurm job submission file:\t"
            f"{job_submission_file.relative_to(os.getcwd())}"
        )

    # scratch = os.environ["SCRATCH"]
    # data_dir = os.path.join(scratch, "/project/LizardLips")

    with open(job_submission_file, "w") as fh:
        fh.writelines("#!/bin/bash\n# -*- coding: utf-8 -*-\nset -euo pipefail\n\n")
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
                'echo - e "SLURM_JOBID:\\t$SLURM_JOBID\n'
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
                "#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery\n\n"
            )

        fh.writelines(
            'echo -e "Resources are allocated."\n'
            f'echo -e "Activating conda environment {conda_env}."\n'
            "source $HOME/miniconda3/etc/profile.d/conda.sh\n"
            f"conda activate {conda_env}\n\n"
        )

        if code.lower() == "dvc":
            if verbose:
                logger = logging.getLogger(__name__)
                logger.info(f"Setting up slurm job submission to run DVC")

            fh.writelines(
                'echo -e "Running DVC repro..."\n'
                f'cd "{Path(input_dir).resolve()}"\n'
                "dvc pull -r origin\n"
                "dvc repro\n\n"
                'echo -e "Exit code:\t$?"'
            )
        elif code.lower() == "jupyter":
            if verbose:
                logger = logging.getLogger(__name__)
                logger.info(
                    f"Setting up slurm job submission to host Jupyter lab "
                    f"listening on port:{port}, ip:{jupy_ip}"
                )

            fh.writelines(
                'echo -e "Running Jupyter Lab...\\n'
                f'\\tListening on port:{port} ip:{jupy_ip}"\n\n'
                f'cd "{Path(input_dir).resolve()}"\n'
                f"jupyter lab --port={port} --ip={jupy_ip}\n\n"
                'echo -e "Exit code:\t$?"'
            )
        else:
            if verbose:
                logger = logging.getLogger(__name__)
                logger.info(f"Setting up slurm job submission to run: " f"{code}")
                fh.writelines(
                    'echo -e "Running code..."\n'
                    f'cd "{Path(input_dir).resolve()}"\n'
                    f"{code}\n\n"
                    'echo -e "Exit code:\t$?"'
                )

    if verbose:
        logger = logging.getLogger(__name__)
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
