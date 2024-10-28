"""
Skeleton for COMP3506/7505 A2, S2, 2024
The University of Queensland
Joel Mackenzie and Vladimir Morozov
"""

from typing import Any
from structures.bit_vector import BitVector

import math

from .util import object_to_byte_array


class BloomFilter:
    """
    A BloomFilter uses a BitVector as a container. To insert a given key, we
    hash the key using a series of h unique hash functions to set h bits.
    Looking up a given key follows the same logic, but only checks if all
    bits are set or not.

    Note that a BloomFilter is considered static. It is initialized with the
    number of total keys desired (as a parameter) and will not grow. You
    must decide what this means in terms of allocating your bitvector space
    accordingly.

    You can add functions if you need to.

    *** A NOTE ON KEYS ***
    We will only ever use int or str keys.
    We will not use `None` as a key.
    You might like to look at the `object_to_byte_array` function
    stored in util.py -- This function can be used to convert a string
    or integer key into a byte array, and then you can use the byte array
    to make your own hash function (bytes are just integers in the range
    [0-255] of course).
    """
    # The probability of a false positive
    fp_r = 0.01

    def __init__(self, max_keys: int) -> None:
        # You should use max_keys to decide how many bits your bitvector
        # should have, and allocate it accordingly.

        self._num_bits = self.__get_bv_size(max_keys)
        self._num_hashes = self.__get_num_hashes(max_keys, self._num_bits)
        # self._num_hashes = 5
        self._data = BitVector()
        self._data.allocate(self._num_bits)
        self._is_empty = True
        self._seeds = self.__get_seeds()

        self._primes = [509, 1021, 2039, 4093,
                        8191, 16381, 32749, 65521, 131071, 262139,
                        524287, 1048573, 2097143, 4194301, 8388593,
                        16777213, 33554393, 67108859, 134217689, 268435399,
                        536870909, 1073741789, 2147483647, 4294967291, 8589934583, 17179869143]

        self._small_primes = [3, 5, 7, 11, 13]

        self._big_prime = self._get_big_prime()
        self._a1 = 5
        self._b1 = 11
        self._a2 = 7
        self._b2 = 13

    def __get_bv_size(self, max_keys: int) -> int:
        m = -(max_keys * math.log(self.fp_r)) / (math.log(2)**2)
        return int(m)

    def __get_num_hashes(self, max_keys: int, num_bits: int) -> int:
        k = (num_bits / max_keys) * math.log(2)
        return int(k)

    def __get_seeds(self):
        seeds = [None] * self._num_hashes

        for i in range(self._num_hashes):
            seeds[i] = i

        return seeds

    def _get_big_prime(self):
        # Select a prime number larger than the number of bits
        i = 0
        while self._primes[i] <= self._num_bits:
            i += 1
        return self._primes[i]

    def __str__(self) -> str:
        """
        A helper that allows you to print a BloomFilter type
        via the str() method.
        This is not marked. <<<<
        """
        print(self._data)

    def __cyclic_shift(self, key: Any) -> int:
        """
        Prehash function: For strings, performs cyclic shift using bytes.
        For integers, returns the number directly.
        """
        if isinstance(key, int):
            pre_hash = key
            pre_hash ^= (pre_hash >> 16)
            pre_hash *= 0x85ebca6b
            pre_hash ^= (pre_hash >> 13)
            pre_hash *= 0xc2b2ae35
            pre_hash ^= (pre_hash >> 16)
            return pre_hash & ((2 ** 32) - 1)

        elif isinstance(key, str):
            byte_array = object_to_byte_array(key)
            littler_prime = 8388593
            pre_hash = 1073741789
            for byte in byte_array:
                # pre_hash = ((pre_hash << 5) | (pre_hash >> (32 - 5))) ^ byte
                # cyclic shift and XOR
                pre_hash ^= byte
                pre_hash = (pre_hash * littler_prime) & ((2 ** 32) - 1)
            return pre_hash

    def __hash1(self, prehash: int) -> int:
        # Hash1 = (a1 * prehash + b1) % p
        return (self._a1 * prehash + self._b1) % self._big_prime

    def __hash2(self, prehash: int) -> int:
        # Hash2 = (a2 * prehash + b2) % p
        return (self._a2 * prehash + self._b2) % self._big_prime

    def insert(self, key: Any, is_kmer: bool = False) -> None:
        """
        Insert a key into the Bloom filter.
        Time complexity for full marks: O(1)
        """
        self._is_empty = False
        prehash = self.__cyclic_shift(key)  # Prehash the key

        hash1_value = self.__hash1(prehash)
        hash2_value = self.__hash2(prehash)

        for i in range(self._num_hashes):
            index = (hash1_value * i + hash2_value) % self._num_bits  # Compute final index
            self._data.set_at(index)

    def contains(self, key: Any, is_kmer: bool = False) -> bool:
        """
        Returns True if all bits associated with the h unique hash functions
        over k are set. False otherwise.
        Time complexity for full marks: O(1)
        """
        prehash = self.__cyclic_shift(key)
        hash1_value = self.__hash1(prehash)
        hash2_value = self.__hash2(prehash)

        for i in range(self._num_hashes):
            index = (hash1_value * i + hash2_value) % self._num_bits
            if not self._data[index]:
                return False
        return True

    def insert_from_kmer_list(self, kmers: list[str]):
        for kmer in kmers:
            self.insert(kmer, True)

    def __contains__(self, key: Any) -> bool:
        """
        Same as contains, but lets us do magic like:
        `if key in my_bloom_filter:`
        Time complexity for full marks: O(1)
        """
        return self.contains(key)

    def is_empty(self) -> bool:
        """
        Boolean helper to tell us if the structure is empty or not
        Time complexity for full marks: O(1)
        """
        return self._is_empty

    def get_capacity(self) -> int:
        """
        Return the total capacity (the number of bits) that the underlying
        BitVector can currently maintain.
        Time complexity for full marks: O(1)
        """
        return self._num_bits
