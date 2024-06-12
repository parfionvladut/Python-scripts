import pandas as pd
import numpy as np
df = pd.read_csv(r'C:\Users\jacso\Desktop\Homework\XLM-USD-Daily-09.15.14-10.30.21.csv', na_values='NaN', keep_default_na=False)
print(np.count_nonzero(df.isnull().values))

