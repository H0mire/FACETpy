import subprocess
import sys

def compile_fastranc():
    # Check if gcc is installed
    try:
        subprocess.run(["gcc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise EnvironmentError("gcc is not installed or not found in PATH")

    if sys.platform.startswith('linux'):
        subprocess.run(["gcc", "-shared", "-fPIC", "-o", "src/facet/helpers/libfastranc.so", "src/facet/helpers/fastranc.c", "-lm"], check=True)
    elif sys.platform.startswith('win'):
        subprocess.run(["gcc", "-shared", "-o", "src/facet/helpers/fastranc.dll", "src/facet/helpers/fastranc.c", "-lm"], check=True)
    else:
        raise OSError("Unsupported operating system")

if __name__ == "__main__":
    compile_fastranc()
