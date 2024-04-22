import numpy as np
from ctypes import CDLL, c_int, c_double, POINTER
import sys
from loguru import logger

# set current path to the path the file is in and then change back to the original path
import os

path = os.path.dirname(os.path.abspath(__file__))


# Erkennen des Betriebssystems
if sys.platform.startswith('linux'):
    lib_path = path + '/libfastranc.so'  # Linux
elif sys.platform.startswith('win'):
    lib_path = path + '/fastranc.dll'  # Windows
else:
    raise OSError("Unsupported operating system")

if os.path.exists(lib_path):
    logger.info(f"File {lib_path} exists")
else:
    logger.error(f"File {lib_path} does not exist")

fastranc = None
try:
    lib = CDLL(lib_path)
    fastranc = lib.fastranc
    fastranc.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        c_int,
        c_double,
        POINTER(c_double),
        POINTER(c_double),
        c_int,
    ]
    fastranc.restype = None
except Exception as ex:
    logger.error(f"File {lib_path} could not be opened! Check your Project Files Message: {ex}")
    logger.info(f"Current working directory: {os.getcwd()}")


def fastr_anc(refs_array, d_array, N_value, mu_value):
    # Umwandeln der numpy Arrays in ctypes und Vorbereiten der Ausgabe-Arrays
    refs_array_ct = np.ctypeslib.as_ctypes(refs_array)
    d_array_ct = np.ctypeslib.as_ctypes(d_array)
    out_array = np.zeros_like(refs_array)
    y_array = np.zeros_like(refs_array)
    veclength_value = len(refs_array)

    # Aufrufen der C-Funktion
    fastranc(
        refs_array_ct,
        d_array_ct,
        N_value,
        mu_value,
        out_array.ctypes.data_as(POINTER(c_double)),
        y_array.ctypes.data_as(POINTER(c_double)),
        veclength_value,
    )

    return out_array, y_array


