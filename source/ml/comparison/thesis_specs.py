import json
import os

import numpy as np

from .metrics import infer_error_mode, parse_variable_name

# default "short list" for thesis plots / summaries
KEY_VARS_DEFAULT = ["th_pt", "tl_pt", "b1_pt", "b2_pt", "ttbar_m"]
QQ_VARS_DEFAULT = ["th_pt", "b1_pt"]

# colors for A vs B comparisons (can override from CLI)
DEFAULT_COLOR_A = "tab:purple"
DEFAULT_COLOR_B = "tab:orange"
DEFAULT_TRUTH_COLOR = "black"

# latex labels for the observable part
OBS_LABELS = {
    "pt": "p_T",
    "eta": "\\eta",
    "phi": "\\phi",
    "m": "m",
}

# particle labels for plotting (also latex-ish)
PARTICLE_LABELS = {
    "th": "t,had",
    "tl": "t,lep",
    "ttbar": "t\\overline{t}",
    "wh": "W,had",
    "wl": "W,lep",
    "b1": "b_1",
    "b2": "b_2",
}

# units to show on the axes
OBS_UNITS = {
    "pt": "GeV",
    "eta": "",
    "phi": "",
    "m": "GeV",
}

# fallback truth ranges if a variable isnt in the json configs
OBS_TRUTH_RANGE_DEFAULTS = {
    "pt": (0.0, 550.0),
    "eta": (-6.0, 7.5),
    "phi": (-3.0, 3.75),
    "m": (0.0, 250.0),
}

# default residual ranges (pt here is fractional resolution)
OBS_RESIDUAL_RANGE_DEFAULTS = {
    "pt": (-1.0, 1.0),
    "eta": (-1.0, 1.0),
    "phi": (-np.pi, np.pi),
}


def _trecnet_root():
    # repo root relative to this file
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_json(path):
    # tiny helper so the main function reads cleaner
    with open(path) as infile:
        return json.load(infile)


def _units_label(units):
    # only show units when they exist (eta/phi dont need clutter)
    if units == "":
        return ""
    return f"[{units}]"


def _variable_label(parton, observable, units):
    # build a latex-ish axis label like $p_T^{t,had}$ [GeV]
    obs_label = OBS_LABELS.get(observable, observable)
    particle_label = PARTICLE_LABELS.get(parton, parton)
    units_label = _units_label(units)
    return f"${obs_label}^{{{particle_label}}}$ {units_label}".rstrip()


def _truth_range(specs, observable, ref_parton=None):
    # reuse ranges from an existing spec if possible
    if ref_parton is not None:
        key = f"{ref_parton}_{observable}"
        if key in specs:
            return specs[key]["truth_range"]
    return OBS_TRUTH_RANGE_DEFAULTS[observable]


def load_thesis_defaults():
    # load plotting defaults from my config jsons + build var_specs dict
    root = _trecnet_root()
    config_dir = os.path.join(root, "config", "plotting", "tommy_plots")

    # example_plot_config has stuff like ATLAS label settings
    plot_cfg = _load_json(os.path.join(config_dir, "example_plot_config.json"))

    # these are the per-variable ranges/binsiwant for truth-reco plots
    truthreco_cfg = _load_json(os.path.join(config_dir, "tommy_truthreco_config.json"))

    # separate config for residual binning
    res_cfg = _load_json(os.path.join(config_dir, "tommy_res_config.json"))

    var_specs = {}

    # build specs for whatever is explicitly in the truthreco config
    for parton, observables in truthreco_cfg["variables"].items():
        for observable, spec in observables.items():
            base_var = f"{parton}_{observable}"
            units = OBS_UNITS.get(observable, "")
            var_specs[base_var] = {
                "variable": base_var,
                "parton": parton,
                "observable": observable,
                "units": units,
                "label": _variable_label(parton, observable, units),
                # truthreco config defines the truth axis range + nbins
                "truth_range": (float(spec["min"]), float(spec["max"])),
                "truth_nbins": int(spec["nbins"]),
                # residual ranges are mostly generic defaults (phi is special)
                "residual_range": OBS_RESIDUAL_RANGE_DEFAULTS.get(observable),
                # residual binning comes from the residual config if available
                "residual_nbins": int(
                    res_cfg["variables"].get(parton, {}).get(observable, {}).get("nbins", 30)
                ),
                # tracks whether residual is plain, wrapped phi, or pt resolution
                "error_mode": infer_error_mode(base_var),
            }

    # for b1/b2 i just reuse ranges from existing particles so it stays consistent
    b_truth_sources = {
        "pt": "th",
        "eta": "th",
        "phi": "th",
        "m": "wh",
    }

    for b_parton in ("b1", "b2"):
        for observable in ("pt", "eta", "phi", "m"):
            base_var = f"{b_parton}_{observable}"
            units = OBS_UNITS[observable]
            var_specs[base_var] = {
                "variable": base_var,
                "parton": b_parton,
                "observable": observable,
                "units": units,
                "label": _variable_label(b_parton, observable, units),
                "truth_range": _truth_range(
                    var_specs,
                    observable,
                    ref_parton=b_truth_sources[observable],
                ),
                # b-jets get a simple default binning for now
                "truth_nbins": 30,
                "residual_range": OBS_RESIDUAL_RANGE_DEFAULTS.get(observable),
                "residual_nbins": 30,
                "error_mode": infer_error_mode(base_var),
            }

    # bundle everything that the report script expects
    return {
        "atlas_label": plot_cfg["atlas_label"],
        "var_specs": var_specs,
        "key_vars_default": KEY_VARS_DEFAULT,
        "qq_vars_default": QQ_VARS_DEFAULT,
        "color_a": DEFAULT_COLOR_A,
        "color_b": DEFAULT_COLOR_B,
        "truth_color": DEFAULT_TRUTH_COLOR,
    }


def get_variable_spec(base_var, defaults):
    # return a copy so callers can tweak ranges without mutating the defaults dict
    var_specs = defaults["var_specs"]
    if base_var in var_specs:
        return dict(var_specs[base_var])

    # make something reasonable if the var isnt in config
    parton, observable = parse_variable_name(base_var)
    units = OBS_UNITS.get(observable, "")
    return {
        "variable": base_var,
        "parton": parton,
        "observable": observable,
        "units": units,
        "label": _variable_label(parton, observable, units),
        "truth_range": OBS_TRUTH_RANGE_DEFAULTS.get(observable, (-1.0, 1.0)),
        "truth_nbins": 30,
        "residual_range": OBS_RESIDUAL_RANGE_DEFAULTS.get(observable),
        "residual_nbins": 30,
        "error_mode": infer_error_mode(base_var),
    }


def freeze_mass_range(error_values):
    # used when we dont have a predefined residual range (mostly for mass-like stuff)
    # this picks a symmetric range based on central 99% so plots arent ruined by outliers.
    error_values = np.asarray(error_values).ravel()
    error_values = error_values[np.isfinite(error_values)]
    if error_values.size == 0:
        return (-250.0, 250.0)

    q_low, q_high = np.quantile(error_values, [0.005, 0.995])
    span = max(abs(q_low), abs(q_high))
    return (-float(span), float(span))