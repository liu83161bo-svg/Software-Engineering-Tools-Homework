# tests/ge_checks.py
import great_expectations as ge
import pandas as pd


def run_ge_checks():

    df = pd.read_csv("./data/sample_eeg_data.csv")


    ge_df = ge.from_pandas(df)


    results = ge_df.expect_table_row_count_to_be_between(min_value=100, max_value=500)
    print(f"Row count check: {results.success}")

    results = ge_df.expect_column_values_to_be_between(
        column="age", min_value=0, max_value=100
    )
    print(f"Age range check: {results.success}")

    return results.success