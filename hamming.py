from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import numpy.random as rn
from itertools import product


alphabet = "abcdefghijklmnopqrstuvwxyz01234567890 .,\n"
digits = "0123456789"

def char2bits(char: chr) -> np.array:
    '''
    Given a character in the alphabet, returns a 8-bit numpy array of 0,1 which represents it
    '''
    num   = ord(char)
    if num >= 256:
        raise ValueError("Character not recognised.")
    bits = format(num, '#010b')
    bits  = [ int(b) for b in bits[2:] ]
    return np.array(bits)

def bits2char(bits) -> chr:
    '''
    Given a 7-bit numpy array of 0,1 or bool, returns the character it encodes
    '''
    bits  = ''.join(bits.astype(int).astype(str))
    num   =  int(bits, base = 2)
    return chr(num)

def text2bits(text: str) -> np.ndarray:
    '''
    Given a string, returns a numpy array of bool with the binary encoding of the text
    '''
    text = text.lower()
    text = [ t for t in text if t in alphabet ]
    bits = [ char2bits(c) for c in text ]
    return np.array(bits, dtype = bool).flatten()

def bits2text(bits: np.ndarray) -> str:
    '''
    Given a numpy array of bool or 0 and 1 (as int) which represents the
    binary encoding a text, return the text as a string
    '''
    if np.mod(len(bits), 8) != 0:
        raise ValueError("The length of the bit string must be a multiple of 8.")
    bits = bits.reshape(int(len(bits)/8), 8)
    chrs = [ bits2char(b) for b in bits ]
    return ''.join(chrs)


#_________________________________________________________________________________________________
#Helper functions
def is_power_of_two(num: int) -> bool:
    '''
    Returns True if num is a power of 2, false otherwise
    '''
    return '1' not in list(bin(num)[3:])

def get_data_indices(m: int) -> np.ndarray:
    '''
    Returns a vector of size 2**m - 1 where the 1s represent the indices of data bits 
    in a hamming code of m parity bits, and the 0s represent the indices of parity bits
    '''
    n = 2**m - 1
    powers_of_two = map(is_power_of_two, (list(range(1,n+1))))
    return (1 - np.array(list(powers_of_two))).astype(bool)

#_________________________________________________________________________________________________
# TASK 1a)
def parity_matrix(m : int) -> np.ndarray:
    """
    m : int
      The number of parity bits to use
    
    return : np.ndarray
      m-by-n parity check matrix
    """

    n = (2**m) - 1 
    p_matrix = np.zeros((m,n)).astype(int) #Create an array with the desired output shape to be filled in

    #By noticing that the column i is the binary number i+1 reversed, we can create the parity check matrix
    for j in range(n):
        binary_rep = np.binary_repr(j+1, m) #Convert to binary of the desired length
        binary_list = list(binary_rep) #Convert binary string to list
        binary_list.reverse()
        p_matrix[:,j] = binary_list
    
    return p_matrix
    
def hamming_generator(m : int) -> np.ndarray:
    """
    m : int
      The number of parity bits to use

    return : np.ndarray
      k-by-n generator matrix
    """
    n = (2**m) - 1 
    k = n - m

    g_matrix = np.zeros(shape=[k,n]).astype(int) #Initialise the generator matrix with desired shape
    p_matrix = parity_matrix(m) #Generate parity matrix

    data_indices = get_data_indices(m)
    data_p_matrix = p_matrix[:,data_indices] #Filter parity matrix to get only data columns

    data_column_counter = 0
    parity_column_counter = 0

    for j in range(n):
        if is_power_of_two(j+1): # If parity column
            g_matrix[:,j] = data_p_matrix[parity_column_counter]
            parity_column_counter += 1
        else: # If data column
            #Add a one at position n of the nth data column 
            #This is equivalent to turning data columns into an identity matrix
            g_matrix[data_column_counter, j] = 1 
            data_column_counter += 1
    return g_matrix



def hamming_encode(data : np.ndarray, m : int) -> np.ndarray:
    """
    data : np.ndarray
      array of shape (k,) with the block of bits to encode

    m : int
      The number of parity bits to use

    return : np.ndarray
      array of shape (n,) with the corresponding Hamming codeword
    """
    assert( data.shape[0] == 2**m - m - 1 )

    generator_matrix = hamming_generator(m)
    return (generator_matrix.T @ data) %2
#_________________________________________________________________________________________________
# TASK 1b)

def hamming_decode(code : np.ndarray, m : int) -> np.ndarray:
    """
    code : np.ndarray
      Array of shape (n,) containing a Hamming codeword computed with m parity bits
    m : int
      Number of parity bits used when encoding

    return : np.ndarray
      Array of shape (k,) with the decoded and corrected data
    """
    assert(np.log2(len(code) + 1) == int(np.log2(len(code) + 1)) == m)

    n = 2**m - 1

    parity_check = (parity_matrix(m) @ code) %2
    dimensions = (len(parity_check),)

    #Find the position of the flipped bit from the parity check (index 0 is position 1)
    flipped_bit = np.sum(2**np.indices(dimensions)*parity_check) 
    if flipped_bit != 0:
        code[flipped_bit-1] = 1 - code[flipped_bit-1]

    data_indices = get_data_indices(m)
    return code[data_indices]


#_________________________________________________________________________________________________
# TASK 1c)

def decode_secret(msg : np.ndarray) -> str:
    """
    msg : np.ndarray
      One-dimensional array of binary integers

    return : str
      String with decoded text
    """
    m = 4 # <-- Your guess goes here
    n = 2**m - 1
    decoded_msg = []

    truncated_length = len(msg) - (len(msg) % n) #Truncate the message so that the last block isn't passed in if it has less bits
    for start in range(0,truncated_length,n):
        decoded_code = hamming_decode(msg[start : start + n], m) #Take a block of n bits to decode
        for bit in decoded_code:
            decoded_msg.append(bit)
    #Truncate the message to the be size of a multiple of 8 to be passed into bits2text
    decoded_msg_mul_8 = np.array(decoded_msg[:(len(decoded_msg) - (len(decoded_msg)%8))]) 
    return bits2text(np.array(decoded_msg_mul_8))

#_________________________________________________________________________________________________
# TASK 2a)
    
def binary_symmetric_channel(data : np.ndarray, p : float) -> np.ndarray:
    """
    data : np.ndarray
      1-dimensional array containing a stream of bits
    p : float
      probability by which each bit is flipped

    return : np.ndarray
      data with a number of bits flipped
    """
    flips = rn.uniform(low = 0,high = 1, size = len(data)) < p #create mask
    flips = flips.astype(int)
    return (data + flips) % 2 #xor-mask

#_________________________________________________________________________________________________
# TASK 2b)

def decoder_accuracy(m : int, p : float) -> float:
    """
    m : int
      The number of parity bits in the Hamming code being tested
    p : float
      The probability of each bit being flipped

    return : float
      The probability of messages being correctly decoded with this
      Hamming code, using the noisy channel of probability p
    """
    n = 2**m - 1
    k = n - m
    correct_decodings = 0
    number_of_trials = 10000
    
    for i in range(number_of_trials):
        # A bit of a cheat but basically each bit starts at 0 and gets flipped with a 50% probability, so the codeword gets randomly generate with uniform probability
        random_codeword = np.random.choice([0,1],size = k)
        encoded_codeword = hamming_encode(random_codeword, m) #Encode
        received_codeword = binary_symmetric_channel(encoded_codeword, p) #Pass through the noisy channel
        decoded_codeword = hamming_decode(received_codeword, m) #Decode
        if (random_codeword == decoded_codeword).all(): #Compare decoded and initial
            correct_decodings += 1
    return(correct_decodings / number_of_trials)
    


if __name__ == '__main__':
  with open('secret.txt', 'r') as f:
      bits = f.read()
  message = np.array([ int(b) for b in bits ]).astype(int)
  print(decode_secret(message))