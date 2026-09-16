from facet import export, load

INPUT_FILE = "./examples/datasets/NiazyFMRI.set"
OUTPUT_FILE = "./examples/datasets/NiazyFMRI.bdf"

ctx = load(INPUT_FILE)
export(ctx, OUTPUT_FILE)
