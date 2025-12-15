import pandas as pd
def missing_values_summarizer(df:pd.DataFrame, 
                              drop_cols:bool=False, 
                              drop_threshold:float=0.1, 
                              verbose=False):

    missing_counts = df.isnull().sum()
    non_missing_counts = df.notnull().sum()
    total_vol = len(df)
    missing_proportions = missing_counts / len(df)
    non_missing_proportions = non_missing_counts / len(df)

    props_df = pd.DataFrame({
        "feature": df.columns,
        "missing_count": missing_counts,
        "non_missing_count": non_missing_counts,
        "total_vol": total_vol,
        "prop_missing": missing_proportions,
        "prop_non_missing": non_missing_proportions,
        'data_type': df.dtypes.values
    })
    props_df.reset_index(inplace=True, drop=True)
    
    if drop_cols:
        cols_to_drop=props_df[props_df['prop_missing'] > drop_threshold]['feature'].tolist()
        df_resized = df.drop(columns=cols_to_drop)
        if verbose:
            print(f'Dropping columns: {cols_to_drop}')
    else:
        df_resized = df.copy()
    
    return props_df, df_resized
