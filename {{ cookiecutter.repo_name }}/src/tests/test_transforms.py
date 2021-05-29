#!/usr/bin/env python3

import numpy as np
import pandas as pd

from src.img import transforms


class TestTransforms:
    def test_zero(self):
        img_array = pd.DataFrame(np.zeros((10, 784), dtype=np.float32))
        mean_image = transforms.mean_image(img_array)
        assert np.mean(mean_image) == 0

    def test_rand(self):
        img_array = pd.DataFrame(np.random.randn(1000, 784), dtype=np.float32)
        mean_image = transforms.mean_image(img_array)
        assert np.subtract(np.mean(mean_image), 0) < 0.005
