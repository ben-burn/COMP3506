"""
Skeleton for COMP3506/7505 A1, S2, 2024
The University of Queensland
Joel Mackenzie and Vladimir Morozov
"""

from typing import Any

from structures.dynamic_array import DynamicArray


class BitVector:
    """
    A compact storage for bits that uses DynamicArray under the hood.
    Each element stores up to 64 bits, making BitVector 64 times more memory-efficient
    for storing bits than plain DynamicArray.
    """

    BITS_PER_ELEMENT = 64

    def __init__(self) -> None:
        """
        We will use the dynamic array as our data storage mechanism
        """
        self._data = DynamicArray()
        self._append = 0
        self._append_len = 0
        self._prepend = 0
        self._prepend_len = 0
        self._bitshift = 0
        self._flipped = False
        self._reversed = False

    def _print(self):
        print(f" append = {self._append}") 
        print(f" prepend = {self._prepend}") 
        print(" data = ", self._data)

    def __str__(self) -> str:
        """
        A helper that allows you to print a BitVector type
        via the str() method.
        """
        pass

    def get_at(self, index: int) -> int | None:
        """
        Get bit at the given index.
        Return None if index is out of bounds.
        Time complexity for full marks: O(1)
        """
        # print('GETTING ITEM')
        # print(f'index = {index}, size = {self.get_size()}')
        # Handle out of bounds
        if not self.__bounds_check(index):
            return 
        
        dy_size = self._data.get_size() * self.BITS_PER_ELEMENT
        # print(f'index = {index}, self._prepend_len = {self._prepend_len}, dy_size = {dy_size}, self._append_len = {self._append_len}')
        # Handle if in self._prepend
        if index < self._prepend_len:
            # print(f'accessing prepend, shift = {self._prepend_len - index - 1}')
            return self.__access_in(self._prepend, self._prepend_len - index - 1)
            
        # check if in main dynamic array
        index -= self._prepend_len
        if index < dy_size:
            # print(f'accessing data, shift = {self.BITS_PER_ELEMENT - position - 1}')
            elem = self._data[index // self.BITS_PER_ELEMENT]
            position = index % self.BITS_PER_ELEMENT
            return self.__access_in(elem, self.BITS_PER_ELEMENT - position - 1)

        # check position in self._append
        index -= dy_size
        # print(f'accessing append, shift = {self._append_len - index - 1}')
        return self.__access_in(self._append, self._append_len - index - 1)
        
    def __getitem__(self, index: int) -> int | None:
        """
        Same as get_at.
        Allows to use square brackets to index elements.
        """
        return self.get_at(index)

    def set_at(self, index: int) -> None:
        """
        Set bit at the given index to 1.
        Do not modify the vector if the index is out of bounds.
        Time complexity for full marks: O(1)
        """
        self.__setitem__(index, 1)

    def unset_at(self, index: int) -> None:
        """
        Set bit at the given index to 0.
        Do not modify the vector if the index is out of bounds.
        Time complexity for full marks: O(1)
        """
        self.__setitem__(index, 0)

    def __setitem__(self, index: int, state: int) -> None:
        """
        Set bit at the given index.
        Treat the integer in the same way Python does:
        if state is 0, set the bit to 0, otherwise set the bit to 1.
        Do not modify the vector if the index is out of bounds.
        Time complexity for full marks: O(1)
        """
        # print("SETTING ITEM")
        # print(f'index = {index}, size = {self.get_size()}')
        if not self.__bounds_check(index):
            return 
        
        state = state if not self._flipped else not state
        dy_size = self._data.get_size() * self.BITS_PER_ELEMENT

        # Handle if in self._prepend
        if index < self._prepend_len:
            # print('accessing prepend')
            mask = self.__create_mask(state, self._prepend_len - index - 1)
            # print(bin(mask))
            if state:
                self._prepend |= mask
            else:
                self._prepend &= mask
            return
            
        # check if in main dynamic array
        index -= self._prepend_len
        if index < dy_size:
            # print('accessing data')
            i = index // self.BITS_PER_ELEMENT
            position = index % self.BITS_PER_ELEMENT
            mask = self.__create_mask(state, self.BITS_PER_ELEMENT - position - 1)
            # print(bin(mask))
            if state:
                self._data[i] |= mask
            else:
                self._data[i] &= mask
            return

        # check position in self._append
        # print('accessing append')
        index -= dy_size
        mask = self.__create_mask(state, self._append_len - index - 1)
        # print(bin(mask))
        if state:
            self._append |= mask
        else:
            self._append &= mask
        return

    def append(self, state: int) -> None:
        """
        Add a bit to the back of the vector.
        Treat the integer in the same way Python does:
        if state is 0, set the bit to 0, otherwise set the bit to 1.
        Time complexity for full marks: O(1*)
        """
        self._append <<= 1
        self._append |= state if not self._flipped else not state
        self._append_len += 1
        if self._append_len == self.BITS_PER_ELEMENT:
            self._data.append(self._append)
            self._append = 0
            self._append_len = 0

    def prepend(self, state: Any) -> None:
        """
        Add a bit to the front of the vector.
        Treat the integer in the same way Python does:
        if state is 0, set the bit to 0, otherwise set the bit to 1.
        Time complexity for full marks: O(1*)
        """
        state = state if not self._flipped else not state
        self._prepend |= (state << self._prepend_len)
        self._prepend_len += 1
        if self._prepend_len == self.BITS_PER_ELEMENT:
            self._data.prepend(self._prepend)
            self._prepend = 0
            self._prepend_len = 0

    def reverse(self) -> None:
        """
        Reverse the bit-vector.
        Time complexity for full marks: O(1)
        """
        self._reversed = not self._reversed

    def flip_all_bits(self) -> None:
        """
        Flip all bits in the vector.
        Time complexity for full marks: O(1)
        """
        self._flipped = not self._flipped

    def shift(self, dist: int) -> None:
        """
        Make a bit shift.
        If dist is positive, perform a left shift by `dist`.
        Otherwise perform a right shift by `dist`.
        Time complexity for full marks: O(N)
        """
        # When bitshift to right need to delete stuff off end of append
        self._bitshift += dist
        if abs(self._bitshift) >= self.BITS_PER_ELEMENT:
            pass

    def rotate(self, dist: int) -> None:
        """
        Make a bit rotation.
        If dist is positive, perform a left rotation by `dist`.
        Otherwise perform a right rotation by `dist`.
        Time complexity for full marks: O(N)
        """

    def get_size(self) -> int:
        """
        Return the number of *bits* in the list
        Time complexity for full marks: O(1)
        """
        return (self._data.get_size() * self.BITS_PER_ELEMENT) + self._append_len + self._prepend_len# + self._bitshift

    def __bounds_check(self, index: int):
        dy_size = self._data.get_size() * self.BITS_PER_ELEMENT
        # print(index, dy_size + self._prepend_len + self._append_len)
        if index < 0 or index >= dy_size + self._prepend_len + self._append_len:
            return False
        return True
    
    def __access_in(self, number: int, shift: int):
        x = (number >> shift) & 1
        return x if not self._flipped else x ^ 1

    def __create_mask(self, state: int, shift: int):
        mask = (1 << shift)
        return mask if state else not mask