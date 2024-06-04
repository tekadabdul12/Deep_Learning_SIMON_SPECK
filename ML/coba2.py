import csv
# mylist = [-1,-2,0,1,2.323e-03]
#
# smallest_value = min(mylist)
# smallest_index = mylist.index(smallest_value)
#
# print(smallest_value,smallest_index)

import pandas as pd

# Membaca file CSV menggunakan pandas
df = pd.read_csv('../output_simon.csv')

# Mengakses kolom yang diperlukan
plaintexts = df['plaintext'].tolist()
ciphers = df['cipher'].tolist()
keys = df['key'].tolist()[:2]

# Mengonversi plaintext dari format heksadesimal ke integer
plaintexts = [int(x, 16) for x in plaintexts]

# Membuat list tuples (plaintext, cipher, key)
# data = list(zip(plaintexts_int, ciphers, keys))

print("tes" + str(2))
