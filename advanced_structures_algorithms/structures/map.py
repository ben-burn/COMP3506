"""
Skeleton for COMP3506/7505 A2, S2, 2024
The University of Queensland
Joel Mackenzie and Vladimir Morozov

Please read the following carefully. This file is used to implement a Map
class which supports efficient insertions, accesses, and deletions of
elements.

There is an Entry type defined in entry.py which *must* be used in your
map interface. The Entry is a very simple class that stores keys and values.
The special reason we make you use Entry types is because Entry extends the
Hashable class in util.py - by extending Hashable, you must implement
and use the `get_hash()` method inside Entry if you wish to use hashing to
implement your map. We *will* be assuming Entry types are used in your Map
implementation.
Note that if you opt to not use hashing, then you can simply override the
get_hash function to return -1 for example.
"""

from typing import Any
from structures.entry import Entry
from structures.dynamic_array import DynamicArray, DumbArray


class Map:
    """
    An implementation of the Map ADT.
    The provided methods consume keys and values via the Entry type.
    """
    # Load at which linear probing becomes inefficient
    resize_load = 0.66
    init_capacity = 157

    def __init__(self) -> None:
        """
        Construct the map.
        You are free to make any changes you find suitable in this function
        to initialise your map.
        """

        self._data = DumbArray(self.init_capacity)

        # Size of the Hash Table
        # self._capacity = self._data.get_capacity()
        self._capacity = 10
        self._size = 0

        # Hash Table sizes
        self._primes = [509, 1021, 2039, 4093,
                        8191, 16381, 32749, 65521, 131071, 262139,
                        524287, 1048573, 2097143, 4194301, 8388593,
                        16777213, 33554393, 67108859, 134217689, 268435399,
                        536870909, 1073741789, 2147483647, 4294967291, 8589934583, 17179869143]

        # For indexing primes array
        self._next_prime = 0
        self._next_prime_max = len(self._primes) - 1

        # Number of hashes entries before load is reached
        self._num_load_cap = int(self.resize_load * self._capacity)

    def __str__(self) -> str:
        return str(self._data)

    def __iter__(self):
        """
        Iterator
        """
        for i in range(self._capacity):
            entry = self._data[i]
            if entry is not None and entry.get_value() is not None:
                yield (entry.get_key(), entry.get_value())

    def __resize(self) -> None:
        """
        If num_load_cap is reached resize the DynamicArray with a new prime as capacity
        """
        # Check if capacity is maxxed out
        if self._next_prime == self._next_prime_max:
            raise Exception("Capacity is maxxed out bro")

        # Get the new capacity
        new_capacity = self._primes[self._next_prime]

        # Create a new dynamic array with the new capacity
        new_data = DumbArray(new_capacity)

        # Rehash all entries from the old map to the new map
        # old_size = self._size
        self._size = 0
        for i in range(self._capacity):
            entry = self._data[i]
            if entry is not None and entry.get_value() is not None:
                self._size += 1
                new_hash = self.__hash(entry, resize=True) % new_capacity
                new_data[new_hash] = entry

        # assert old_size == self._size

        # Set the new map and capacity
        self._data = new_data
        self._capacity = new_capacity

        # Increment the prime array index
        self._next_prime += 1

        # Update the new number of entries before sub-optimal load
        self._num_load_cap = int(self.resize_load * self._capacity)

    def insert(self, entry: Entry) -> Any | None:
        """
        Associate value v with key k for efficient lookups. If k already exists
        in your map, you must return the old value associated with k. Return
        None otherwise. (We will not use None as a key or a value in our tests).
        Time complexity for full marks: O(1*)
        """
        # Get the hash of the key
        hash = self.__hash(entry)

        # Probe to find if the key exists
        # if the key exists return the old_value, update to new value
        # else place entry at new index
        exists, index = self.__probe(hash, entry)
        if exists:
            ret_val = self._data[index].get_value()
            self._data[index].update_value(entry.get_value())
            return ret_val

        if self._data[index] is None or self._data[index].get_value() is None:
            self._size += 1

        self._data[index] = entry

        # Increment size and check load
        # self._size += 1
        if self._size >= self._num_load_cap:
            self.__resize()

        return None

    def __hash(self, entry: Entry, seed: int = 1, resize: bool = False) -> int:
        """
        Hashes the an entry key
        """
        universe = (2 ** 32) - 1
        base = 31
        pre_hash = seed
        key = entry.get_key()

        if isinstance(key, str):
            for i, char in enumerate(key):
                num = ord(char)

                # Mix bits more effectively by varying shift amount and using bitwise XOR and OR
                shift_amount = (i * 7) % 32
                pre_hash ^= (num << shift_amount) | (num >> (32 - shift_amount))
                pre_hash = (pre_hash * base + num + seed)

        elif isinstance(key, int):
            pre_hash = (seed * base * key)

        if resize:
            return (pre_hash % universe)
        return (pre_hash % universe) % self._capacity
        # print(entry.get_key() % self._capacity)
        # return entry.get_key() % self._capacity

    def __probe(self, ptr: int, entry: Entry) -> tuple[bool, int]:
        """
        If the probe finds a matching key, and the value is not none, return (True, index)
        Else return (False, available_index)
        """
        prev_grave = None

        while True:
            map_entry = self._data[ptr]

            # If the slot is empty, return the index (or the first prev_grave if found)
            if map_entry is None:
                if prev_grave is not None:
                    # print("returning pounter to first grave")
                    return (False, prev_grave)
                # print("spot is none")
                return (False, ptr)

            # If it's a prev_grave (key exists but value is None), store prev_grave
            if map_entry.get_value() is None:
                # print("found grave - setting prev_grave")
                if prev_grave is None:
                    prev_grave = ptr

            # If we find a matching key with a valid value, return True and the index
            elif map_entry == entry:
                # print("found entry - updating")
                return (True, ptr)

            # Move to the next slot (linear probing)
            # print("probing")
            ptr = (ptr + 1) % self._capacity

    def insert_kv(self, key: Any, value: Any) -> Any | None:
        """
        A version of insert which takes a key and value explicitly.
        Handy if you wish to provide keys and values directly to the insert
        function. It will return the value returned by insert, so keep this
        in mind. You can modify this if you want, as long as it behaves.
        Time complexity for full marks: O(1*)
        """
        new_entry = Entry(key, value)
        return self.insert(new_entry)

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        For convenience, you may wish to use this as an alternative
        for insert as well. However, this version does _not_ return
        anything. Can be used like: my_map[some_key] = some_value
        Time complexity for full marks: O(1*)
        """
        self.insert(Entry(key, value))

    def remove(self, key: Any) -> None:
        """
        Remove the key/value pair corresponding to key k from the
        data structure. Don't return anything.
        Time complexity for full marks: O(1*)
        """
        dummy_entry = Entry(key, None)
        hash = self.__hash(dummy_entry)
        # Check if key exists
        exists, index = self.__probe(hash, dummy_entry)

        # If key exists, create a grave
        if exists:
            self._data[index] = dummy_entry
            self._size -= 1

    def find(self, key: Any) -> Any | None:
        """
        Find and return the value v corresponding to key k if it
        exists; return None otherwise.
        Time complexity for full marks: O(1*)
        """
        dummy_entry = Entry(key, None)
        hash = self.__hash(dummy_entry)
        # Check if key exists
        exists, index = self.__probe(hash, dummy_entry)

        # If key exists, return value
        if exists:
            return self._data[index].get_value()
        return None

    def __getitem__(self, key: Any) -> Any | None:
        """
        For convenience, you may wish to use this as an alternative
        for find()
        Time complexity for full marks: O(1*)
        """
        return self.find(key)

    def get_size(self) -> int:
        """
        Time complexity for full marks: O(1)
        """
        return self._size

    def is_empty(self) -> bool:
        """
        Time complexity for full marks: O(1)
        """
        return not self._size

    def to_list(self) -> list:
        entries = [None] * self._size
        ptr = 0
        for i in range(self._capacity):
            if self._data[i] is not None and self._data[i].get_value() is not None:
                entries[ptr] = self._data[i]
                ptr += 1
        return entries


class Set(Map):
    def insert(self, element: Any) -> None:
        super().insert(Entry(element, 1))
