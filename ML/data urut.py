data_A = [
    0x1410, 0x1420, 0x1430, 0x1440, 0x1450,
    0x1460, 0x1470, 0x1480, 0x1490, 0x14a0,
    0x14b0, 0x14c0, 0x14d0, 0x14e0, 0x14f0,
    0x1500, 0x1510, 0x1520, 0x1530, 0x1540,
    0x1550, 0x1560, 0x1570, 0x1580, 0x1590, 0x15a0

]

data_B = [
    0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A,
    0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53, 0x54,
    0x55, 0x56, 0x57, 0x58, 0x59, 0x5A
]

data_2D = list(zip(data_A, data_B))
data_hex = [(hex(a),hex(b)) for a, b in data_2D]

#data_int = [(int(a, 16), int(b, 16)) for a, b in data_hex]

print(data_hex)

for i in range(26):
    print('0x1918111009080100',end=",")