#!/usr/bin/env python3

import os
import click
import cv2
from dotenv import find_dotenv, load_dotenv
import logging
import numpy as np
from pathlib import Path
import pandas as pd
import skimage.io as io

# import cv2
# import numpy as np
# import pandas as pd
# import skimage.io as io

import tensorflow as tf
from tensorflow.data import AUTOTUNE


# load custom libraries from src
from src.data import mapfile, load_data
from src.img.tfreader import tf_imread, tf_imread_predict, tf_imreadpair, tf_dataset
from src.img.augment import apply_transforms, tf_standardize, tf_normalize
from src.data import load_params
from src.models import networks

# set tf warning options
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def predict(
    mapfile_path,
    model_path,
    params_filepath="params.yaml",
    results_dir="./results",
    debug=False,
):
    """Accept filepaths to the mapfile, load model and predict ouputs."""
    assert type(mapfile_path) is str, TypeError(f"MAPFILE must be type STR")

    # start logger
    logger = logging.getLogger(__name__)

    # read files
    mapfile_df = load_data(mapfile_path, sep=",", header=0, index_col=0, dtype=str)

    # load model
    model = tf.keras.models.load_model(model_path)

    # load params
    params = load_params(params_filepath)
    train_params = params["train_model"]
    random_seed, target_size, n_classes, mean_img, std_img = (
        params["random_seed"],
        params["target_size"],
        params["n_classes"],
        params["mean_img"],
        params["std_img"],
    )
    batch_size, epochs = train_params["batch_size"], train_params["epochs"]

    # create dataset using tf.data
    data_records = [list(elem) for elem in mapfile_df.to_records(index=False)]
    dataset = tf.data.Dataset.from_tensor_slices(data_records)

    # build tf.data pipeline
    if params["segmentation"]:
        dataset = dataset.map(
            tf_imread_predict,
            num_parallel_calls=AUTOTUNE,
        )
    else:
        dataset = dataset.map(
            tf_imread,
            num_parallel_calls=AUTOTUNE,
        )

    # pre-processing
    dataset = dataset.map(
        lambda x: (tf_standardize(x, mean=mean_img, std=std_img)),
        num_parallel_calls=AUTOTUNE,
    )

    # # pre-processing
    # dataset = dataset.map(
    #     lambda x, y: (
    #         tf_standardize(x, mean=mean_img, std=std_img),
    #         y,
    #     ),
    #     num_parallel_calls=AUTOTUNE,
    # )

    dataset = dataset.cache().batch(  # cache after mapping
        batch_size=batch_size,
        drop_remainder=True,
        # num_parallel_calls=AUTOTUNE,
        deterministic=debug,
    )

    # set output filepath
    if not Path(results_dir).exists:
        print("Directory not exists!")
        return
    
    img_filepath = Path(results_dir).joinpath("test_image.png")

    # test filepath
    Path("nerve_0000.png").resolve()

    test_img = dataset.take(60)
    img_predict = model.predict(dataset)
    for idx in range(mapfile_df.shape[0]):
        temp_img = img_predict[idx]
        # img_filepath = Path("./results").joinpath(f"test_image_{idx:03d}.png")
        img_filepath = Path("./results").joinpath(
            mapfile_df["filename"][idx].replace(".png", "_mask.png")
        )
        cv2.imwrite(str(img_filepath), (temp_img[:, :, 1] * 255).astype(np.uint8))
        # print(f"min={np.min(img_predict[:])}\nmax={np.max(img_predict[:])}")

    # # predict single image
    # test_img = dataset.take(1)
    # img_predict = model.predict(test_img)[0]
    # print(f"min={np.min(img_predict[:])}\nmax={np.max(img_predict[:])}")
    # img_predict = (img_predict[:,:,1] * 255).astype(np.uint8)
    # cv2.imwrite(str(img_filepath), img_predict)
    # # norm = np.zeros(img_predict.shape)
    # # img_predict2 = cv2.normalize(img_predict, norm, 0, 255, cv2.NORM_MINMAX).astype(
    # #     np.uint8
    # # )
    # # cv2.imwrite(str(img_filepath), img_predict2)
    # # cv2.imwrite(str(img_filepath).replace("test", "orig"), (test_img * 255).astype(np.uint8))
    return True


@click.command()
@click.argument(
    "mapfile_path",
    default=Path("./data/processed/mapfile_df.csv").resolve(),
    type=click.Path(exists=True),
    # help="Filepath to the CSV with image filenames and class labels.",
)
@click.argument(
    "model_path",
    type=click.Path(exists=True),
)
@click.option("--params_filepath", "-p", default="params.yaml")
# @click.option(
#     "--results-dir",
#     default=Path("./results").resolve(),
#     type=click.Path(),
# )
@click.option(
    "--debug",
    is_flag=True,
    help="Debug switch that turns off augmentation, shuffle, and makes runs deterministic.",
)
def main(
    mapfile_path,
    model_path,
    params_filepath="params.yaml",
    results_dir="./results",
    debug=False,
):
    predict(
        mapfile_path,
        model_path,
        params_filepath=params_filepath,
        results_dir=results_dir,
        debug=debug,
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
