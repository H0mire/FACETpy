import subprocess
import sys

def compile_fastranc():
    if sys.platform.startswith('linux'):
        subprocess.run(["gcc", "-shared", "-fPIC", "-o", "src/facet/helpers/libfastranc.so", "src/facet/helpers/fastranc.c", "-lm"], check=True)
    elif sys.platform.startswith('win'):
        subprocess.run(["gcc", "-shared", "-o", "src/facet/helpers/fastranc.dll", "src/facet/helpers/fastranc.c", "-lm"], check=True)
    else:
        raise OSError("Unsupported operating system")
