import pandas as pd
import ast

def parse_list_col(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):  
        return x
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []