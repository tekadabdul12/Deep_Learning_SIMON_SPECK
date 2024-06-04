import pandas as pd

df = pd.read_csv("../tes.csv")

print(df)

df.rename(columns={"nama_kolom_a" : "kolom1"}, inplace=True)
print(df)

x = df['kolom1'].values.tolist()

print(x)