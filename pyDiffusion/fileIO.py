import numpy as np
import npquad
import csv


def loadArrayQuad(fileName, delimiter=",", dtype=np.quad, skiprows=0):
    """
    Load a quad array from a file.

    Parameters
    ----------
    fileName : str
        name of file to read

    delimiter : str
        Delimiter used to save the file

    dtype : type
        Type of the output. Constructor must accept strings

    skiprows : int
        Number of rows to skip at the beginning of the file.

    Returns
    -------
    numpy array
        File loaded as a numpy array

    Note
    ----
    Takes about 20 seconds to save/load an array of size 1e7.
    """
    with open(fileName, "r") as file:
        for _ in range(skiprows):
            file.readline()
        line = file.__next__()
        array = np.empty(
            (1 + sum(1 for line in file), len(line.split(delimiter))), dtype=dtype
        )

    with open(fileName, "r") as file:
        for _ in range(skiprows):
            file.readline()
        for i, line in enumerate(file):
            for j, element in enumerate(line.split(delimiter)):
                array[i, j] = dtype(element)
    return array


def saveArrayQuad(save_file, arr):
    """
    Save a numpy array of quads to a file.

    Parameters
    ----------
    save_file : str
        File to save array to

    arr : numpy array (dtype np.quad)
        Array to save
    """

    if isinstance(arr, list):
        arr = np.array(arr, dtype=np.quad)

    with open(save_file, "w+") as f:
        writer = csv.writer(f)
        if arr.ndim == 1:
            writer.writerow(arr)
        else:
            for row in range(arr.shape[0]):
                writer.writerow(arr[row, :])
