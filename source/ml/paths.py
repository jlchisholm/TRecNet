############################################################################
# Path management for models, outputs, and logs
# - uses environment variables for root overrides
# - supports legacy and new directory layouts
# - to use: import paths; paths.resolve_model_dir(...)
# - it will identify existing model dirs by model_id
# - locate the model dir with resolve_model_dir(model_id)
# Author: Tommy Lubomirski
# 26th September 2025
############################################################################

import glob
import os

# define environment variables (assumes you are working inside TRecNet)
DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT = os.environ.get("TRECNET_ROOT", DEFAULT_ROOT)
# where to output/retrive trained models, outputs, and tensorboard logs
TRAINED_ROOT = os.environ.get("TRECNET_TRAINED_ROOT", os.path.join(ROOT, "trained_models"))
OUTPUTS_ROOT = os.environ.get("TRECNET_OUTPUTS_ROOT", os.path.join(ROOT, "outputs"))
TB_ROOT = os.environ.get("TRECNET_TB_ROOT", os.path.join(ROOT, "tensorboard_logs"))


def ensure_dir(path):
    """Ensure that a directory exists. Create it if it doesn't."""
    os.makedirs(path, exist_ok=True)
    return path


# this is a generator function, so it yields candidates one by one
# it 'yields' an iterable of possible model directories
def model_dir_candidates(model_id):
    '''Yield possible model directory candidates for a given model_id.'''
    # Direct access: trained_models/<model_id>/
    yield os.path.join(TRAINED_ROOT, model_id)
    # One-level nested access: trained_models/<family>/<model_id>/
    for d in glob.glob(os.path.join(TRAINED_ROOT, "*", model_id)):
        yield d
    # Nested access: trained_models/<model_name>/<model_id>/
    for d in glob.glob(os.path.join(TRAINED_ROOT, "*", "*", model_id)):
        yield d
    # Higher-level nested access: trained_models/<sub-dir>/<model_name>/<model_id>
    for d in glob.glob(os.path.join(TRAINED_ROOT, "*", "*", "*", model_id)):
        yield d

def resolve_model_dir(model_id: str) -> str:
    """ main entry point to resolve a model directory by model_id """

    # identify the model dir by checking candidates
    for d in model_dir_candidates(model_id):
        if os.path.isdir(d):
            #return the first match
            return d
    raise FileNotFoundError(f"The model dir for '{model_id}' under {TRAINED_ROOT} could not be found.")

def model_run_dir(model_version: str, model_name: str, model_id: str) -> str:
    '''This function ensures backward compatibility with the old layout, if its not working, look here'''
    # use the nested layout
    d = os.path.join(TRAINED_ROOT, model_version, model_name, model_id)
    return ensure_dir(d)

def model_subdir(model_dir: str, *parts) -> str:
    '''Create a subdirectory within the model directory.'''
    return ensure_dir(os.path.join(model_dir, *parts))

def model_file(model_dir: str, fname: str) -> str:
    '''Create a file path within the model directory.'''
    return os.path.join(model_dir, fname)

def model_paths(model_version: str, model_name: str, model_id: str):
    '''Return a dictionary of important paths for a given model.'''
    base = model_run_dir(model_version, model_name, model_id)
    return {
        "base": base,
        "model":    model_subdir(base, "model"),
        "info":     model_subdir(base, "info"),
        "history":  model_subdir(base, "history"),
        "plots":    model_subdir(base, "plots"),
        "plots_train": model_subdir(base, "plots", "train"),
        "plots_val_scaled": model_subdir(base, "plots", "val", "scaled"),
        "plots_val_original": model_subdir(base, "plots", "val", "original"),
        "scaling":  model_subdir(base, "scaling"),
    }

def keras_path(model_dir: str, model_id: str) -> str:
    ''' This was added to support keras model saving/loading
     (couldnt get unfreezing to work otherwise) '''
    return os.path.join(model_dir, "model", f"{model_id}.keras")

def tb_fit_dir(model_id: str) -> str:
    ''' Directory for TensorBoard fit logs for a given model_id.
    added to ensure backward compatibility (again, atleast i think so) '''
    return ensure_dir(os.path.join(TB_ROOT, "fit", model_id))

def outputs_dir(model_id: str, mode: str, dataset_basename: str) -> str:
    ''' Directory for outputs of a given model_id and mode (train/val/test). '''
    # dataset_basename without ".h5", assumes dataset_basename is a filename
    out = os.path.join(OUTPUTS_ROOT, model_id, mode, dataset_basename)
    return ensure_dir(out)
