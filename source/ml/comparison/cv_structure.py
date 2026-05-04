##########################################################################
#                                                                        #
#  cv_structure.py                                                       #
#  Author: Tommy Lubomirski.                                             #
#  Created: Nov. 27/25                                                   #
#                                                                        #
#  Helpers for finding CV folds and loading the ROOT outputs.            #
#                                                                        #
##########################################################################

import os
import re
import logging
import uproot
import awkward as ak
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _find_fold_name(path_parts):
    # walk the path tokens and grab the first fold_* segment we see
    for part in path_parts:
        if part.startswith("fold_"):
            return part
    return None


def parse_fold_name(fold_name):
    # fold naming convention: fold_r{rep}_f{fold}
    match = re.fullmatch(r"fold_r(\d+)_f(\d+)", fold_name)
    if match is None:
        # if this triggers, something upstream changed the folder naming
        raise RuntimeError(f"Fold name does not match expected pattern: {fold_name}")
    return int(match.group(1)), int(match.group(2))


def _sorted_fold_items(fold_to_value):
    # sort like r0_f0, r0_f1, r1_f0
    return sorted(fold_to_value.items(), key=lambda item: parse_fold_name(item[0]))


def _list_fold_root_files(cv_dir, results_subdir="results", name_substring=None):
    # crawl a *_CV directory and find the Results_*.root for each fold
    cv_dir = os.path.abspath(cv_dir)
    if not os.path.isdir(cv_dir):
        raise RuntimeError(f"Cv directory does not exist: {cv_dir}")

    fold_to_file = {}

    for dirpath, _, filenames in os.walk(cv_dir):
        parts = dirpath.split(os.sep)

        # only care about folders that look like .../fold_*/results/...
        if results_subdir not in parts:
            continue

        fold_name = _find_fold_name(parts)
        if fold_name is None:
            logger.debug(
                f"results_subdir found in {dirpath}, but no 'fold*' segment; skipping."
            )
            continue

        # grab all root files in this results directory
        root_files = [f for f in filenames if f.endswith(".root")]
        if name_substring is not None:
            # optional filter if you want a specific Results file
            root_files = [f for f in root_files if name_substring in f]

        if not root_files:
            logger.debug(f"No ROOT files found in {dirpath} matching criteria.")
            continue

        # if there are multiple just pick the first alphabetically
        chosen = sorted(root_files)[0]
        full_path = os.path.join(dirpath, chosen)

        # dont silently overwrite if we somehow find multiple results dirs per fold
        if fold_name in fold_to_file:
            logger.warning(
                f"Multiple ROOT files found for {fold_name}; "
                f"keeping first seen: {fold_to_file[fold_name]} (skipping {full_path})"
            )
            continue

        fold_to_file[fold_name] = full_path
        logger.info(f"Discovered fold '{fold_name}' -> {full_path}")

    if not fold_to_file:
        # if this happens, the directory structure probably changed
        raise RuntimeError(
            f"No ROOT files found under CV dir '{cv_dir}' "
            f"with results_subdir='{results_subdir}' and "
            f"name_substring={name_substring!r}."
        )

    return fold_to_file


def get_fold_to_file(cv_dir, results_subdir="results", name_substring=None):
    # public helper: returns {fold_name: /abs/path/to/Results.root} sorted by fold
    return dict(
        _sorted_fold_items(
            _list_fold_root_files(
                cv_dir=cv_dir,
                results_subdir=results_subdir,
                name_substring=name_substring,
            )
        )
    )


def load_truth_reco_df(root_path, base_vars=None, include_event_number=True):
    # load one fold ROOT file and return a single DataFrame with truth_* and reco_* columns
    root_path = os.path.abspath(root_path)
    logger.info(f"Loading ROOT file: {root_path}")

    with uproot.open(root_path) as f:
        # sanity check: we expect exactly these trees from the TRecNet outputs
        if "parton" not in f or "reco" not in f:
            raise RuntimeError(
                f"ROOT file {root_path} missing required 'parton' or 'reco' trees.\n"
                f"but found: {list(f.keys())}"
            )

        # awkward -> pandas is convenient here, and fast enough for what we need
        parton_arr = f["parton"].arrays(library="ak")
        reco_arr = f["reco"].arrays(library="ak")

        truth_df = ak.to_dataframe(parton_arr)
        truth_df = truth_df.add_prefix("truth_")

        # keep eventNumber around if it exists (helps for debugging / cross-checks)
        if "truth_eventNumber" in truth_df.columns:
            truth_df = truth_df.rename(columns={"truth_eventNumber": "eventNumber"})

        reco_df = ak.to_dataframe(reco_arr)
        reco_df = reco_df.add_prefix("reco_")

        # merge side by side
        df = pd.concat([truth_df, reco_df], axis=1)

    #  restrict to a smaller set of variables so we dont carry huge tables around
    if base_vars is not None:
        keep_cols = []

        if include_event_number and "eventNumber" in df.columns:
            keep_cols.append("eventNumber")

        for var in base_vars:
            truth_col = f"truth_{var}"
            reco_col = f"reco_{var}"
            if truth_col in df.columns:
                keep_cols.append(truth_col)
            if reco_col in df.columns:
                keep_cols.append(reco_col)

        # warn if we asked for things that arent there
        missing = [
            var for var in base_vars
            if f"truth_{var}" not in df.columns and f"reco_{var}" not in df.columns
        ]
        if missing:
            logger.warning(
                f"Some requested base_vars not found in {root_path}: {', '.join(missing)}"
            )

        if keep_cols:
            df = df[keep_cols]
        else:
            # if we filtered everything out by accident, just return full df
            logger.warning(f"No requested variables found in {root_path}; returning full DataFrame.")

    return df.reset_index(drop=True)


def load_all_folds_as_dfs(
    cv_dir,
    results_subdir="results",
    name_substring=None,
    base_vars=None,
    include_event_number=True,
):
    # load every fold in a CV directory into memory
    fold_to_file = get_fold_to_file(
        cv_dir=cv_dir,
        results_subdir=results_subdir,
        name_substring=name_substring,
    )

    fold_to_df = {}
    for fold, path in _sorted_fold_items(fold_to_file):
        df = load_truth_reco_df(
            path,
            base_vars=base_vars,
            include_event_number=include_event_number,
        )
        fold_to_df[fold] = df
        logger.info(f"Loaded fold '{fold}' with {len(df)} events")

    return fold_to_df


def pooled_truth_reco(fold_to_df, base_var):
    # pool truth/reco arrays across all folds
    truth_col = f"truth_{base_var}"
    reco_col = f"reco_{base_var}"

    truth_chunks = []
    reco_chunks = []

    for _, df in _sorted_fold_items(fold_to_df):
        # if a fold is missing this var, just skip it
        if truth_col not in df.columns or reco_col not in df.columns:
            logger.warning(
                "Missing %s or %s in a fold when pooling %s; skipping that fold.",
                truth_col, reco_col, base_var
            )
            continue

        y_true = df[truth_col].to_numpy()
        y_pred = df[reco_col].to_numpy()

        # keep only events where both are finite
        mask = np.isfinite(y_true) & np.isfinite(y_pred)

        truth_chunks.append(y_true[mask])
        reco_chunks.append(y_pred[mask])

    if not truth_chunks:
        # nothing found (or everything was NaN) -> return empty arrays
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.concatenate(truth_chunks), np.concatenate(reco_chunks)