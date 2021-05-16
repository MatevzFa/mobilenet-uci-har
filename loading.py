from pathlib import Path
import numpy as np
from collections import defaultdict


base = Path(
    "/home/matevz/coding/MAG/HAR-pipline/Batch/Data/Original-Data/UCI-HAR-Dataset/Inertial-Signals")


def load(id, axis, mode, dtype):
    path = base / mode / f"{id}_{axis}_{mode}.txt"
    return np.loadtxt(path, dtype=dtype)


def compose(mode, dtype):
    """
    Loads and reshapes np array of shape
    [N, 3, 3, 128]
     |  |  |   |
     +- number of inputs
        |  |   |
        +- 3 sensors
           |   |
           +- 3 axes
               |
               +- 128 samples in segment

    Into a [N, 3, 32, 32] matrix

    Returns the reshaped matrix
    """
    sensors = ["body_acc", "body_gyro", "total_acc"]
    axes = ["x", "y", "z"]
    d = defaultdict(lambda: dict())
    for id in sensors:
        for axis in axes:
            d[id][axis] = load(id, axis, mode, dtype=dtype)

    n, w = d[sensors[0]][axes[0]].shape

    out_axes = ["x", "y", "z", "y", "x", "z", "x", "y"]

    data = np.empty((n, len(sensors), len(out_axes), w))
    desired_w = 32

    if w % desired_w != 0:
        raise Exception("sizes not matching")

    for i, id in enumerate(sensors):
        for c, axis in enumerate(out_axes):
            data[:, i, c, :] = d[id][axis]

    data_out = np.empty((n, len(sensors), desired_w, desired_w), dtype=dtype)
    data_out[:, :, 0:8, :] = data[:, :, :, 0:32]
    data_out[:, :, 8:16, :] = data[:, :, :, 32:64]
    data_out[:, :, 16:24, :] = data[:, :, :, 64:96]
    data_out[:, :, 24:32, :] = data[:, :, :, 96:128]

    return data_out
