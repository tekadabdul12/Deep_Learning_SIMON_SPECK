from simon import SimonCipher

key =  0x1918111009080100
plaintxt = 0x65656877
ciphertxt = 0xc69be9bb
block_size = 32
key_size = 64
c = SimonCipher(key, key_size, block_size)

chipher = c.encrypt(plaintxt)

de = c.decrypt(ciphertxt)


print(hex(chipher))
print(type(chipher))
print(hex(de))
