import pandas as pd  # type: ignore

from sklearn.model_selection import train_test_split  # type: ignore

from pathlib import Path
import argparse
import logging


def clean(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy(deep=True)
    clean_df.loc[clean_df["title"].str.split().str.len() == 0, "title"] = (
        pd.NA
    )  # Empty title is same as NaN
    clean_df.loc[clean_df["text"].str.split().str.len() == 0, "text"] = (
        pd.NA
    )  # Empty text is same as NaN
    clean_df = clean_df.dropna(axis=0, how="any")
    return clean_df


def preprocess_data(data_dir: str | Path, test_size: float):
    data_dir = Path(data_dir)
    logging.info("Reading original data from directory...")
    data = pd.read_csv(data_dir / "WELFake_Dataset.csv", index_col=0)

    logging.info("Cleaning data...")
    clean_df = clean(data)

    logging.info("Train/test splitting data...")
    train_df, test_df = train_test_split(clean_df, test_size=test_size, random_state=42)

    logging.info("Persisting resulting dataframes...")
    train_df.to_csv(data_dir / "WELFake_clean_train.csv", index=True)
    test_df.to_csv(data_dir / "WELFake_clean_test.csv", index=True)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        help="Name of directory where WELFake_Dataset.csv is located.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Number in range (0, 1) denoting proportion of dataset to leave for test",
    )
    parsed = parser.parse_args()

    preprocess_data(data_dir=parsed.data_dir, test_size=parsed.test_size)
