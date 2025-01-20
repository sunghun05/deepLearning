import pandas as pd
df = pd.read_csv('gdp_pcap.csv', sep='\t')

print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())

# types in pandas| in python
# object         | string
# int64          | int
# float64        | float
# datetime64     | datetime


