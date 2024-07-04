from __future__ import print_function
import csv


class SpeckCipher(object):
    """Speck Block Cipher Object"""
    # valid cipher configurations stored:
    # block_size:{key_size:number_rounds}
    __valid_setups = {32: {64: 22},
                      48: {72: 22, 96: 23},
                      64: {96: 26, 128: 27},
                      96: {96: 28, 144: 29},
                      128: {128: 32, 192: 33, 256: 34}}

    __valid_modes = ['ECB', 'CTR', 'CBC', 'PCBC', 'CFB', 'OFB']

    def encrypt_round(self, x, y, k):
        """Complete One Round of Feistel Operation"""
        rs_x = ((x << (self.word_size - self.alpha_shift)) + (x >> self.alpha_shift)) & self.mod_mask

        add_sxy = (rs_x + y) & self.mod_mask

        new_x = k ^ add_sxy

        ls_y = ((y >> (self.word_size - self.beta_shift)) + (y << self.beta_shift)) & self.mod_mask

        new_y = new_x ^ ls_y

        return new_x, new_y

    def decrypt_round(self, x, y, k):
        """Complete One Round of Inverse Feistel Operation"""

        xor_xy = x ^ y

        new_y = ((xor_xy << (self.word_size - self.beta_shift)) + (xor_xy >> self.beta_shift)) & self.mod_mask

        xor_xk = x ^ k

        msub = ((xor_xk - new_y) + self.mod_mask_sub) % self.mod_mask_sub

        new_x = ((msub >> (self.word_size - self.alpha_shift)) + (msub << self.alpha_shift)) & self.mod_mask

        return new_x, new_y

    def __init__(self, key, key_size=128, block_size=128, mode='ECB', init=0, counter=0):

        # Setup block/word size
        try:
            self.possible_setups = self.__valid_setups[block_size]
            self.block_size = block_size
            self.word_size = self.block_size >> 1
        except KeyError:
            print('Invalid block size!')
            print('Please use one of the following block sizes:', [x for x in self.__valid_setups.keys()])
            raise

        # Setup Number of Rounds and Key Size
        try:
            self.rounds = self.possible_setups[key_size]
            self.key_size = key_size
        except KeyError:
            print('Invalid key size for selected block size!!')
            print('Please use one of the following key sizes:', [x for x in self.possible_setups.keys()])
            raise

        # Create Properly Sized bit mask for truncating addition and left shift outputs
        self.mod_mask = (2 ** self.word_size) - 1

        # Mod mask for modular subtraction
        self.mod_mask_sub = (2 ** self.word_size)

        # Setup Circular Shift Parameters
        if self.block_size == 32:
            self.beta_shift = 2
            self.alpha_shift = 7
        else:
            self.beta_shift = 3
            self.alpha_shift = 8

        # Parse the given iv and truncate it to the block length
        try:
            self.iv = init & ((2 ** self.block_size) - 1)
            self.iv_upper = self.iv >> self.word_size
            self.iv_lower = self.iv & self.mod_mask
        except (ValueError, TypeError):
            print('Invalid IV Value!')
            print('Please Provide IV as int')
            raise

        # Parse the given Counter and truncate it to the block length
        try:
            self.counter = counter & ((2 ** self.block_size) - 1)
        except (ValueError, TypeError):
            print('Invalid Counter Value!')
            print('Please Provide Counter as int')
            raise

        # Check Cipher Mode
        try:
            position = self.__valid_modes.index(mode)
            self.mode = self.__valid_modes[position]
        except ValueError:
            print('Invalid cipher mode!')
            print('Please use one of the following block cipher modes:', self.__valid_modes)
            raise

        # Parse the given key and truncate it to the key length
        try:
            self.key = key & ((2 ** self.key_size) - 1)
        except (ValueError, TypeError):
            print('Invalid Key Value!')
            print('Please Provide Key as int')
            raise

        # Pre-compile key schedule
        self.key_schedule = [self.key & self.mod_mask]
        l_schedule = [(self.key >> (x * self.word_size)) & self.mod_mask for x in
                      range(1, self.key_size // self.word_size)]

        for x in range(self.rounds - 1):
            new_l_k = self.encrypt_round(l_schedule[x], self.key_schedule[x], x)
            l_schedule.append(new_l_k[0])
            self.key_schedule.append(new_l_k[1])

        # generate All Key
        global keyall
        keyall = self.key_schedule
        print("AlL Round Key")
        i = 1
        for x in keyall:
            print(i, '.', hex(x))
            i += 1
        print('')

    def encrypt(self, plaintext):
        try:
            b = (plaintext >> self.word_size) & self.mod_mask
            a = plaintext & self.mod_mask
        except TypeError:
            print('Invalid plaintext!')
            print('Please provide plaintext as int')
            raise

        if self.mode == 'ECB':
            b, a = self.encrypt_function(b, a)

        elif self.mode == 'CTR':
            true_counter = self.iv + self.counter
            d = (true_counter >> self.word_size) & self.mod_mask
            c = true_counter & self.mod_mask
            d, c = self.encrypt_function(d, c)
            b ^= d
            a ^= c
            self.counter += 1

        elif self.mode == 'CBC':
            b ^= self.iv_upper
            a ^= self.iv_lower
            b, a = self.encrypt_function(b, a)

            self.iv_upper = b
            self.iv_lower = a
            self.iv = (b << self.word_size) + a

        elif self.mode == 'PCBC':
            f, e = b, a
            b ^= self.iv_upper
            a ^= self.iv_lower
            b, a = self.encrypt_function(b, a)
            self.iv_upper = (b ^ f)
            self.iv_lower = (a ^ e)
            self.iv = (self.iv_upper << self.word_size) + self.iv_lower

        elif self.mode == 'CFB':
            d = self.iv_upper
            c = self.iv_lower
            d, c = self.encrypt_function(d, c)
            b ^= d
            a ^= c
            self.iv_upper = b
            self.iv_lower = a
            self.iv = (b << self.word_size) + a

        elif self.mode == 'OFB':
            d = self.iv_upper
            c = self.iv_lower
            d, c = self.encrypt_function(d, c)
            self.iv_upper = d
            self.iv_lower = c
            self.iv = (d << self.word_size) + c

            b ^= d
            a ^= c

        ciphertext = (b << self.word_size) + a

        return ciphertext

    def decrypt(self, ciphertext):
        try:
            b = (ciphertext >> self.word_size) & self.mod_mask
            a = ciphertext & self.mod_mask
        except TypeError:
            print('Invalid ciphertext!')
            print('Please provide plaintext as int')
            raise

        if self.mode == 'ECB':
            b, a = self.decrypt_function(b, a)

        elif self.mode == 'CTR':
            true_counter = self.iv + self.counter
            d = (true_counter >> self.word_size) & self.mod_mask
            c = true_counter & self.mod_mask
            d, c = self.encrypt_function(d, c)
            b ^= d
            a ^= c
            self.counter += 1

        elif self.mode == 'CBC':
            f, e = b, a
            b, a = self.decrypt_function(b, a)
            b ^= self.iv_upper
            a ^= self.iv_lower

            self.iv_upper = f
            self.iv_lower = e
            self.iv = (f << self.word_size) + e

        elif self.mode == 'PCBC':
            f, e = b, a

            b, a = self.decrypt_function(b, a)

            b ^= self.iv_upper
            a ^= self.iv_lower
            self.iv_upper = (b ^ f)
            self.iv_lower = (a ^ e)
            self.iv = (self.iv_upper << self.word_size) + self.iv_lower

        elif self.mode == 'CFB':
            d = self.iv_upper
            c = self.iv_lower
            self.iv_upper = b
            self.iv_lower = a
            self.iv = (b << self.word_size) + a
            d, c = self.encrypt_function(d, c)

            b ^= d
            a ^= c

        elif self.mode == 'OFB':
            d = self.iv_upper
            c = self.iv_lower
            d, c = self.encrypt_function(d, c)

            self.iv_upper = d
            self.iv_lower = c
            self.iv = (d << self.word_size) + c

            b ^= d
            a ^= c

        plaintext = (b << self.word_size) + a

        return plaintext

    def encrypt_function(self, upper_word, lower_word):

        x = upper_word
        y = lower_word
        a = 1
        global listy
        listy = []

        # Run Encryption Steps For Appropriate Number of Rounds
        for k in self.key_schedule:
            rs_x = ((x << (self.word_size - self.alpha_shift)) + (x >> self.alpha_shift)) & self.mod_mask

            add_sxy = (rs_x + y) & self.mod_mask

            x = k ^ add_sxy

            ls_y = ((y >> (self.word_size - self.beta_shift)) + (y << self.beta_shift)) & self.mod_mask

            y = x ^ ls_y

            z = str(hex(x)[2:]) + str(hex(y)[2:])
            print(a ,z) #on for see all round
            a +=1
            listy.append(z)


        return x, y

        # generate Round (spesific round)
    def hasilround(self, param, round=None):
        a = 1
        listx = []

        if param == True:
            # print('cipher for round', round, )
            print(listy[round])  # spesific round
            listx.append(listy[round])
        elif param == False:
            print("round enkripsi :")  # all round
            for i in listy:
                print(a, i)
                listx[a] = i
                a += 1

        # print(listy)
        return listx

    def decrypt_function(self, upper_word, lower_word):

        x = upper_word
        y = lower_word
        a =1

        # Run Encryption Steps For Appropriate Number of Rounds
        for k in reversed(self.key_schedule):
            xor_xy = x ^ y

            y = ((xor_xy << (self.word_size - self.beta_shift)) + (xor_xy >> self.beta_shift)) & self.mod_mask

            xor_xk = x ^ k

            msub = ((xor_xk - y) + self.mod_mask_sub) % self.mod_mask_sub

            x = ((msub >> (self.word_size - self.alpha_shift)) + (msub << self.alpha_shift)) & self.mod_mask

            z = str(hex(x)[2:]) + str(hex(y)[2:])
            print(a ,z)
            a += 1

        return x, y

    def update_iv(self, new_iv=None):
        if new_iv:
            try:
                self.iv = new_iv & ((2 ** self.block_size) - 1)
                self.iv_upper = self.iv >> self.word_size
                self.iv_lower = self.iv & self.mod_mask
            except TypeError:
                print('Invalid Initialization Vector!')
                print('Please provide IV as int')
                raise
        return self.iv


if __name__ == "__main__":

    #to show all round see line 278

    data = []
    # Membaca file CSV
    with open('input.csv', mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row["Plaintext"])  # Menambahkan elemen kolom "Nama" ke list

    print("Plaintext:", data)
    # cipher = SpeckCipher(0x1918111009080100, 64, 32, 'ECB')
    # g = cipher.encrypt(0x6574694c)
    # print("\ndekripsi\n")
    # d = cipher.decrypt(0xa86842f2)
    # print("\nEnkripsi")
    # print(hex(g))
    # print("Deksripsi")
    # print(hex(d))

    for j in range(5):
        with open('output_speck' + str(j) + '.csv', mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(['plaintext', 'cipher', 'key'])

            # writer.writeheader()  # Menulis
            for i, value in enumerate(data):
                # print("\n",i)
                w = SpeckCipher(0x1918111009080100, key_size=64, block_size=32)
                t = w.encrypt((int(value, 16)))  # round final
                round = w.hasilround(True, j)  # round pilihan
                writer.writerow([value, round[0], hex(keyall[j])])



