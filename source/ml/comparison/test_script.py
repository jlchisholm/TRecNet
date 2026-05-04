if __name__ == "__main__":
    """
    Minimal sanity check on a *single* ROOT file:
      - Print file keys
      - Print basic info for 'parton' and 'reco' trees
      - Convert to a DataFrame and show a small summary
    """
    import logging
    import uproot
    from source.ml.comparison.cv_structure import load_truth_reco_df
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    root_path = (
        '/data/tommylub/TRecNet_outputs/trained_models/TRecNet_ttbb_v5/TRecNet_ttbb_v5_b1b2+ttbar/TRecNet_ttbb_v5_b1b2+ttbar_10jets_20251128_203433_CV/folds/fold_r0_f0/results/Results_r0_f0.root'
)

    print("\n=== Inspecting single ROOT file ===")
    print("File:", root_path)

    with uproot.open(root_path) as f:
        print("Keys in file:", list(f.keys()))

        for tree_name in ["parton", "reco"]:
            if tree_name in f:
                tree = f[tree_name]
                branches = list(tree.keys())
                print(f"\nTree: {tree_name}")
                print("  n_entries:", tree.num_entries)
                print("  n_branches:", len(branches))
                print("  First ~15 branches:", branches[:15])
            else:
                print(f"\nTree: {tree_name} NOT FOUND in file.")


    print("\n=== Converting single ROOT to DataFrame (truth+reco) ===")
    start_time = time.time()
    df = load_truth_reco_df(root_path)
    end_time = time.time()
    print(f"Conversion took {end_time - start_time:.2f} seconds.")
    print("DataFrame shape:", df.shape)
    print("First 10 columns:", df.columns[:10].tolist())
    print("\nDataFrame head():")
    print(df.head())