import numpy as np
from ctypes import CDLL, c_int, c_double, POINTER

#set current path to the path the file is in and then change back to the original path
import os
path = os.path.abspath(os.getcwd())
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Laden der kompilierten Shared Library
lib_path = './fastranc.dll'  # Stellen Sie sicher, dass dieser Pfad korrekt ist
lib = CDLL(lib_path)

# Definieren des Funktionsprototyps in Python
# Hier nehmen wir an, dass die Funktion in der C-Bibliothek `fastranc_wrapper` heißt,
# passen Sie den Namen entsprechend an, falls er anders ist.
fastranc = lib.fastranc
fastranc.argtypes = [POINTER(c_double), POINTER(c_double), c_int, c_double, POINTER(c_double), POINTER(c_double), c_int]
fastranc.restype = None

def fastr_anc(refs_array, d_array, N_value, mu_value):
    # Umwandeln der numpy Arrays in ctypes und Vorbereiten der Ausgabe-Arrays
    refs_array_ct = np.ctypeslib.as_ctypes(refs_array)
    d_array_ct = np.ctypeslib.as_ctypes(d_array)
    out_array = np.zeros_like(refs_array)
    y_array = np.zeros_like(refs_array)
    veclength_value = len(refs_array)

    # Aufrufen der C-Funktion
    fastranc(refs_array_ct, d_array_ct, N_value, mu_value,
             out_array.ctypes.data_as(POINTER(c_double)),
             y_array.ctypes.data_as(POINTER(c_double)),
             veclength_value)
    
    return out_array, y_array

# Change back to the original path
os.chdir(path)
