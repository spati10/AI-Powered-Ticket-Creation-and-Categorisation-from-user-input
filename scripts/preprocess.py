import pandas as pd
from clean_text import clean_text

def preprocess_dataframe(df, text_col):
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)
    return df
