from pathlib import Path
import pandas as pd

def explore_data(df):


    missing_values_count = df.isna().sum()
    dup_values_count = df.duplicated().sum()
    categories_parts = df["category"].value_counts(normalize=True) * 100
    priority_parts = df["priority"].value_counts(normalize=True) * 100
    mean_text = df["text"].apply(
        lambda x: len(x)
    ).mean()
    most_famous_category = df["category"].max()

    print("numbers of columns", len(df.columns))
    print("numbers of rows", len(df))
    print("data types\n", df.dtypes)
    print("numbers of missing values\n", missing_values_count)
    print("numbers of dup values:", dup_values_count)
    print("categories parts\n", categories_parts)
    print("priority parts\n", priority_parts)
    print("length mean text:", mean_text)
    print("most famous category: ", most_famous_category)

def clean_dataset(df):
    white_list_categories = [
        "network",
        "access",
        "software",
        "security"
    ]
    df.drop_duplicates(inplace=True)
    df["category"].dropna(inplace=True)
    df["text"].dropna(inplace=True)
    df["category"] = df["category"].apply(
        lambda x: str.lower(x).strip()
    )
    state_categories_is_valid=df["category"].isin(white_list_categories).array

    categories_is_invalid = state_categories_is_valid.__contains__(False)

    if categories_is_invalid == True:
        raise Exception("some categories are invalid")

    df["text_length"] = df["text"].apply(
        lambda x: len(x)
    )
    print("cleaning_df\n",df)
    return  df
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data/tickets.csv"
    df = pd.read_csv(csv_path)
    print("=====Explore Data=====")
    explore_data(df)
    print("====Clean Data========")
    clean_df = clean_dataset(df)

    clean_tickets_path = project_root / "data/clean_tickets.csv"
    clean_df.to_csv(clean_tickets_path,index=False)

