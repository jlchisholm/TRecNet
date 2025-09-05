output_path=${1:-TRecNetContainer.sif}
def_file=${2:-trecnetContainer.def}
singularity build ${output_path} ${def_file}