import pandas as pd  # type: ignore

from sklearn.model_selection import train_test_split  # type: ignore

from pathlib import Path
import argparse
import logging


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy(deep=True)
    clean_df.loc[clean_df["title"].str.split().str.len() == 0, "title"] = (
        pd.NA
    )  # Empty title is same as NaN
    clean_df.loc[clean_df["text"].str.split().str.len() == 0, "text"] = (
        pd.NA
    )  # Empty text is same as NaN
    clean_df = clean_df.dropna(axis=0, how="any")
    return clean_df


def preprocess_dataset(
    orig_df: pd.DataFrame, test_size: float, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Cleaning data...")
    clean_df = clean_dataset(orig_df)

    logging.info("Train/test splitting data...")
    train_df, test_df = train_test_split(
        clean_df, test_size=test_size, random_state=random_state
    )

    return train_df, test_df


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

    data_dir = Path(parsed.data_dir)
    logging.info("Reading original data from directory...")
    orig_df = pd.read_csv(data_dir / "WELFake_Dataset.csv", index_col=0)

    train_df, test_df = preprocess_dataset(orig_df=orig_df, test_size=parsed.test_size)

    logging.info("Persisting resulting dataframes...")
    train_df.to_csv(data_dir / "WELFake_clean_train.csv", index=True)
    test_df.to_csv(data_dir / "WELFake_clean_test.csv", index=True)
