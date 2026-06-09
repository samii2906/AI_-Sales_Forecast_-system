import pandas as pd
import numpy as np

def preprocess_dataframe(df):
    """Auto-detect and clean dataframe"""
    report = {}
    
    # Missing values
    missing = df.isnull().sum()
    report['missing_before'] = missing[missing > 0].to_dict()
    
    # Fill numeric with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    report['missing_after'] = df.isnull().sum().sum()
    
    # Detect date columns
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    # Duplicates
    dupes = df.duplicated().sum()
    report['duplicates_removed'] = dupes
    df.drop_duplicates(inplace=True)
    
    return df, report

def detect_column_types(df):
    """Return dict of column roles"""
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include='object').columns.tolist()
    datetime = df.select_dtypes(include='datetime').columns.tolist()
    return {'numeric': numeric, 'categorical': categorical, 'datetime': datetime}