from dataset_builder.build_dataset import build_dataset

if __name__ == "__main__":
    build_dataset(
        top_pct=0.05,
        mid_pct=0.05,
        bottom_pct=0.05,
        out_csv="data/labeled_pairs_3class.csv"
    )
