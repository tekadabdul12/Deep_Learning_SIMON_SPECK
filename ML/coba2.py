import csv
# mylist = [-1,-2,0,1,2.323e-03]
#
# smallest_value = min(mylist)
# smallest_index = mylist.index(smallest_value)
#
# print(smallest_value,smallest_index)

data_input = []
with open('../input.csv', mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data_input.append(row["Plaintext"])  # Menambahkan elemen kolom "Nama" ke list

data_input = [int(x, 16) for x in data_input]

print(data_input)