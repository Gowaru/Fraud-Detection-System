#!/usr/bin/env python3
"""
Script de validation des données avec Great Expectations
"""

import great_expectations as ge
import pandas as pd
import sys

def validate_data(data_path):
    """Valide les données avec Great Expectations"""
    
    # Chargement des données
    df = pd.read_csv(data_path)
    
    # Conversion en GE DataFrame
    ge_df = ge.from_pandas(df)
    
    # Définition des expectations
    ge_df.expect_column_values_to_not_be_null('amount')
    ge_df.expect_column_values_to_be_between('amount', min_value=0, max_value=100000)
    ge_df.expect_column_values_to_be_in_set('fraud', [0, 1])
    ge_df.expect_column_values_to_be_between('repeat_retailer', min_value=0, max_value=1)
    ge_df.expect_table_row_count_to_be_between(min_value=100, max_value=1000000)
    
    # Validation
    results = ge_df.validate()
    
    if results['success']:
        print("✅ Data validation successful")
        return 0
    else:
        print("❌ Data validation failed")
        print(results)
        return 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "data/transactions.csv"
    
    sys.exit(validate_data(data_path))