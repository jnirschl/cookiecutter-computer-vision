#!/usr/bin/env python3

import logging
import os
from pathlib import Path

import click
import cv2
import numpy as np
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv
from MightyMosaic import MightyMosaic

# load custom libraries from src
from src.data import load_params

# set tf warning options
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def predict(
    input_dir: str,
    model_path: str,
    params_filepath: str = "params.yaml",
    results_dir: str = "./results",
    file_ext: str = "png",
    overlap_factor: int = 2,
    batch_size: int = 2,
):
    """Accept input directory and trained model, and save predicted output images."""

    # resolve directories
    input_dir = Path(input_dir).resolve()
    model_path = Path(model_path).resolve()
    results_dir = Path(results_dir).resolve()

    assert input_dir.exists(), NotADirectoryError(f"{input_dir}")
    assert model_path.exists(), NotADirectoryError(f"{model_path}")

    # start logger
    logger = logging.getLogger(__name__)

    # load model
    model = tf.keras.models.load_model(model_path)

    # load params
    params = load_params(params_filepath)
    target_size, mean_img, std_img = (
        params["target_size"],
        params["mean_img"],
        params["std_img"],
    )

    file_list = input_dir.glob("*." + file_ext.replace(".", ""))
    for file in file_list:
        # logger.info(f"Found {len(list(file_list))} images.")
        temp_filepath = Path(file).resolve()
        if temp_filepath.is_file():
            img = cv2.imread(str(temp_filepath), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_std = np.divide(np.subtract(img / 255, mean_img), std_img)

            mosaic = MightyMosaic.from_array(
                img_std, target_size[0:2], overlap_factor=overlap_factor
            )
            prediction = mosaic.apply(
                model.predict, progress_bar=True, batch_size=batch_size
            )
            pred_img = prediction.get_fusion()[:, :, 1]
            pred_img = cv2.normalize(pred_img, None, 0, 255, cv2.NORM_MINMAX)

            out_filepath = str(results_dir.joinpath(file.name))
            cv2.imwrite(out_filepath, pred_img)

    return True


@click.command()
@click.argument(
    "input_dir",
    default=Path("./data/raw/data").resolve(),
    type=click.Path(exists=True),
    # help="Filepath to the CSV with image filenames and class labels.",
)
@click.argument(
    "model_path", type=click.Path(exists=True),
)
@click.option("--params_filepath", "-p", default="params.yaml")
@click.option("--file_ext", default="png")
@click.option("--overlap_factor", default=4)
@click.option("--batch_size", "-b", default=16)
@click.option(
    "--results-dir", "-r", default=Path("./results").resolve(), type=click.Path(),
)
def main(
    input_dir,
    model_path,
    params_filepath="params.yaml",
    results_dir="./results",
    file_ext="png",
    overlap_factor=2,
    batch_size=8,
):
    predict(
        input_dir,
        model_path,
        params_filepath=params_filepath,
        results_dir=results_dir,
        file_ext=file_ext,
        overlap_factor=overlap_factor,
        batch_size=batch_size,
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
