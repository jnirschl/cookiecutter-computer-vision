import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

# load custom libraries from src
from src.data import load_params, mapfile, save_params
from src.img import compute_mean

# set tf warning options
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create(
    input_dir: str,
    output_dir: str,
    output_filename: str = "mapfile_df.csv",
    params_filepath: str = "params.yaml",
    force: bool = True,
    na_rep: str = "nan",
) -> pd.DataFrame:
    """Create mapfile and train/validation splits from directories of image.

    Runs data processing scripts to accept a directory INPUT_DIR containing
    sub-folders of images with one sub-folder per class and create a CSV file
    mapping the image filepath to integer class labels, which is saved in
    OUTPUT_DIR.

    Args:
        input_dir: str with full path to input directory
        output_dir: str with full path to output directory
        output_filename: str with desired output name for mapfile
        params_filepath: str with filename for parameter YAML, which must be in project root
        force: bool to force overwrite pre-existing files
        na_rep: str with value to replace nan values in

    Returns:
        Pandas Dataframe with image filepath and class for classification or
        image filepath and mask filepath for segmentation
    """
    # fix output_filename file extension if not given
    output_filename = Path(output_filename)
    if output_filename.suffix.lower() != ".csv":
        output_filename = output_filename.stem + ".csv"

    # load params_filepath
    params = load_params(params_filepath)
    img_shape = tuple(params["target_size"])
    grayscale = params["color_mode"].lower() == "grayscale"

    # create mapfile_df
    mapfile_df = mapfile.create(
        input_dir,
        output_dir,
        output_filename=output_filename,
        na_rep=na_rep,
        segmentation=params["segmentation"],
        save_format=params["save_format"],
    )
    mapfile_path = str(Path(output_dir).joinpath(output_filename).resolve())

    # compute mean image
    mean_img, std_img = compute_mean.image(
        mapfile_path,
        img_shape=img_shape,  # TODO
        grayscale=grayscale,
        force=force,
        # params_filepath=params_filepath,
    )

    #
    params["mean_img"] = [
        float(elem) for elem in np.mean(mean_img, axis=tuple(range(2))) / 255.0
    ]
    params["std_img"] = [
        float(elem) for elem in np.mean(std_img, axis=tuple(range(2))) / 255.0
    ]
    # save parameters
    save_params(params, filepath=params_filepath)

    # split train test dev
    mapfile.split(mapfile_df, output_dir=output_dir, params_filepath=params_filepath)
    return mapfile_df


@click.command()
@click.argument(
    "input_dir",
    default=Path("./data/raw").resolve(),
    type=click.Path(exists=True),
)
@click.argument(
    "output_dir",
    default=Path("./data/processed").resolve(),
    type=click.Path(exists=True),
)
@click.argument("output_filename", default="mapfile_df.csv", type=click.Path())
@click.option("--params_filepath", "-p", default="params.yaml")
@click.option(
    "--force",
    is_flag=True,
    help="Switch to force overwrite existing mean image.",
)
@click.option("--na-rep", default="nan")
def main(
    input_dir,
    output_dir,
    output_filename="mapfile_df.csv",
    params_filepath="params.yaml",
    force=True,
    na_rep="nan",
):
    create(
        input_dir,
        output_dir,
        output_filename=output_filename,
        params_filepath=params_filepath,
        force=force,
        na_rep=na_rep,
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
