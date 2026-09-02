# utils/misc.py
# Some helper functions

import pandas as pd
import ast
import numpy as np

# Turns collected string data into lists
# Useful for genres, themes, and demographics
def parse_list_col(x):
  if isinstance(x, (list, tuple, np.ndarray)):
    return list(x)

  if pd.isna(x) or x is None:
    return []

  if isinstance(x, str):
    x = x.strip()
    if not x:
      return []
    try:
      parsed = ast.literal_eval(x)
      return (
          parsed
          if isinstance(parsed, list)
          else [parsed]
          if parsed is not None
          else []
      )
    except (ValueError, SyntaxError):
      return []

  return [x]