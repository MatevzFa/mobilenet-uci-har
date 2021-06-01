"""
This script is used to transform UCI-HAR intertial signals into binary
data files used by ApproxTuner for determining NN configurations at 
different points on the tradeoff curve.

To run this script, the following environment variables have to be set:

    HAR_PIPELINE_PATH           must point to the root of
                                https://github.com/davors/HAR-pipline

    HPVM_MODEL_PARAMS_PATH      must point to 'model_params' directory where 
                                ApproxTuner expects tuning and test datasets. 
                                See 
                                https://hpvm.readthedocs.io/en/latest/getting-started.html
                                for further information.
"""

import loading
import numpy as np
from pathlib import Path
import os


def env(key):
    val = os.getenv(key)
    if val is None or len(val) == 0:
        raise RuntimeError(f"Enviromnent variable {key} must be provided")
    return val


def save(arr: np.ndarray, base: Path, path: str):
    arr.tofile(base / path)


def parts(arr, size):
    return arr[:size], arr[size:2*size]


def main():

    har_data_path = Path(env("HAR_PIPELINE_PATH")) / \
        "Batch/Data/Original-Data/UCI-HAR-Dataset/Processed-Data"

    y_train_txt = har_data_path / ".." / "y_train.txt"
    y_test_txt = har_data_path / ".." / "y_test.txt"

    train_data = loading.compose("train", np.float32)
    test_data = loading.compose("test", np.float32)

    train_labels = np.loadtxt(y_train_txt, dtype=np.int32)
    test_labels = np.loadtxt(y_test_txt, dtype=np.int32)

    full_input = np.concatenate([train_data, test_data], axis=0)
    full_labels = np.concatenate([train_labels, test_labels], axis=0)

    assert len(full_input) == len(full_labels)
    permutation = np.random.permutation(len(full_input))

    full_input[:, :, :, :] = full_input[permutation, :, :, :]
    full_labels[:] = full_labels[permutation]

    print(f"full_data={full_input.shape}")
    print(f"full_labels={full_labels.shape}")

    n = 5000
    tune_input, test_input = parts(full_input, n)
    tune_labels, test_labels = parts(full_labels, n)

    print(f"tune_input={tune_input.shape}")
    print(f"test_input={test_input.shape}")
    print(f"tune_labels={tune_labels.shape}")
    print(f"test_labels={test_labels.shape}")

    bin_base = Path(env("HPVM_MODEL_PARAMS_PATH")) / "mobilenet_uci-har"

    save(tune_input, bin_base, "tune_input.bin")
    save(test_input, bin_base, "test_input.bin")
    save(tune_labels, bin_base, "tune_labels.bin")
    save(test_labels, bin_base, "test_labels.bin")


if __name__ == '__main__':
    main()
