import subprocess
import sys
from pathlib import Path


def compile_fastranc():
    module_dir = Path(__file__).resolve().parent
    helpers_dir = module_dir / "helpers"
    source_file = helpers_dir / "fastranc.c"

    if not source_file.exists():
        raise OSError(f"fastranc source file not found: {source_file}")

    # Check if gcc is installed
    try:
        subprocess.run(["gcc", "--version"], check=True, capture_output=True)
    except FileNotFoundError as err:
        raise OSError("gcc is not installed or not found in PATH") from err

    if sys.platform.startswith("linux"):
        output_file = helpers_dir / "libfastranc.so"
        subprocess.run(
            [
                "gcc",
                "-shared",
                "-fPIC",
                "-o",
                str(output_file),
                str(source_file),
                "-lm",
            ],
            check=True,
        )
    elif sys.platform.startswith("win"):
        output_file = helpers_dir / "fastranc.dll"
        subprocess.run(
            ["gcc", "-shared", "-o", str(output_file), str(source_file), "-lm"],
            check=True,
        )
    elif sys.platform.startswith("darwin"):
        output_file = helpers_dir / "libfastranc.dylib"
        subprocess.run(
            ["gcc", "-dynamiclib", "-o", str(output_file), str(source_file), "-lm"],
            check=True,
        )
    else:
        raise OSError("Unsupported operating system")


if __name__ == "__main__":
    compile_fastranc()
