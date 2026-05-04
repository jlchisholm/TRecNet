# compute the stats needed for batch sampler

import argparse
import json
import os
from typing import Dict, List

import h5py
import numpy as np
import tqdm

def parse_args():

    # command line arguments
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--prefix', default='dataset')

    default_vars = [
        'ttbar_m', 'ttbar_pt', 'ttbar_eta',
        'th_pt', 'th_m', 'th_eta',
        'tl_pt', 'tl_m', 'tl_eta'
    ]

    p.add_argument('--vars', nargs='+', default=default_vars)

    p.add_argument('--quantiles', nargs='+', type=float,
                   default=[0.5, 0.8, 0.9, 0.95, 0.99, 0.999])
    
    p.add_argument('--bin_quantiles', nargs='+', type=float,
                   default=[0.8, 0.95, 0.99, 1.0])
    p.add_argument('--max-samples', type=int, default=None)

    # give seed option for reproducability
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# read array from h5 file, make sure its numpy 1d
def get_array(h5, key):
    if key not in h5:
        return []
    
    arr = h5[key][:]

    # make sure its numpy 1d array
    arr = np.array(arr).reshape(-1)
    return arr


# get summary stats from the passed array, return dict, with primitive types
# so it behaves well with json
def summarize_array(a, q_list):
    q_vals = np.quantile(a, q_list)
    
    stats = {
        'count': int(a.size),
        'min': float(np.min(a)),
        'max': float(np.max(a)),
        'mean': float(np.mean(a)),
        'std': float(np.std(a)),
        'quantiles': {f'{q:g}': float(v) for q, v in zip(q_list, q_vals)}
    }

    return stats

def main():
    args = parse_args()

    # make file path
    os.makedirs(args.outdir, exist_ok=True)

    # open file safely
    with h5py.File(args.input, 'r') as f:

        # subsampling index 
        idx = None
        # if max samples is specified, randomly sample that many entries
        if args.max_samples is not None:
            rng = np.random.default_rng(args.seed)
            # use the first var to determine size of dataset

            first_var = args.vars[0]
            n_total = len(f[first_var])

            if args.max_samples > n_total:
                raise ValueError(f'Cannot sample {args.max_samples} from {n_total}')
            
            idx = rng.choice(n_total, size=args.max_samples, replace=False)
            idx.sort()

        # collect per variable stats

        report = {
            'input': args.input,
            'variables': []
        }

        sampler_edges = {}
        means = {}
        stds = {}

        for var in tqdm.tqdm(args.vars):
            arr = get_array(f, var)

            # subsample if needed
            if idx is not None:
                arr = arr[idx]

            s = summarize_array(arr, args.quantiles)

            q_edges = np.quantile(arr, args.bin_quantiles)

            sampler_edges[var] = q_edges.astype(np.float64)

            means[var] = float(np.mean(arr))
            stds[var] = float(np.std(arr, ddof=0))

            report['variables'].append({
                'name': var,
                **s,
                'bin_quantiles': {f'{q:g}': float(v) for q, v in zip(args.bin_quantiles, q_edges)}
            })

        # write to a json file

        json_path = os.path.join(args.outdir, f'{args.prefix}_stats.json')
        with open(json_path, 'w') as fp:
            json.dump(report, fp, indent=2)
        print(f'Wrote stats to {json_path}')

        # we also want save this to a npz file for easy loading in batch sampler

        npz_dict = {
            'vars': np.array(args.vars, dtype=object),
            "bin_quantiles": np.array(args.bin_quantiles, dtype=np.float64),
            "means": np.array([means[v] for v in args.vars], dtype=np.float64),
            "stds": np.array([stds[v] for v in args.vars], dtype=np.float64),
        }

        for v in args.vars:
            npz_dict[f'edges_{v}'] = np.asarray(sampler_edges[v], dtype=np.float64)

        npz_path =  os.path.join(args.outdir, f'{args.prefix}_sampler.npz')
        np.savez_compressed(npz_path, **npz_dict)
        print(f'Wrote sampler edges to {npz_path}')

if __name__ == '__main__':
    main()

