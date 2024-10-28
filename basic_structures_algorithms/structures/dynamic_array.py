"""
Skeleton for COMP3506/7505 A1, S2, 2024
The University of Queensland
Joel Mackenzie and Vladimir Morozov
"""

from typing import Any

class DynamicArray:
    def __init__(self) -> None:
        self._size = 0
        self._capacity = 4
        self._data = [None] * self._capacity
        # The next position available for prepending
        self._start = self._capacity - 1
        # The next position available for appending
        self._end = 0
        # The nth element of the array to return
        self._current = 0
        self._reversed = False

    def bv_init(self, n) -> None:
        BITS_PER_ELEMENT = n
        self._size = int((2**32) / BITS_PER_ELEMENT)
        self._capacity = self._size
        self._data = [0] * self._size
        self._start = self._capacity - 1
        self._end = 0
        self._current = 0
        self._reversed = False

    def __str__(self) -> str:
        """
        A helper that allows you to print a DynamicArray type
        via the str() method.
        """
        if not self._size:
            return "[]"
        list_str = "["
        # mid = f"{self._data[index]}, "
        # last = f"{self._data[index]}"
        # for i in range(self._size):
        #     j = (self._start + i) % self._capacity
        #     if i != self._size - 1:
        #         list_str += f"{self._data[j]}, "
        #     else:
        #         list_str += f"{self._data[j]}"
        for i, x in enumerate(self):
            if i != self._size - 1:
                list_str += f"{x}, "
            else:
                list_str += f"{x}]"
        return list_str

    def __iter__(self) -> None:
        return self
    
    def __next__(self) -> Any:
        if self._current == self._size:
            self._current = 0
            raise StopIteration
        rev = -1 if self._reversed else 1
        index = (self._start + (rev) * (1 + self._current)) % self._capacity
        current_element = self._data[index]
        self._current += 1
        return current_element

    def __resize(self) -> None:
        new_data = [None] * (self._capacity * 2)
        # for i in range(self._size):
            # j = (self._start + i + 1) % self._capacity
            # new_data[i] = self._data[j]
        for i, x in enumerate(self):
            new_data[i] = x
        self._capacity *= 2
        self._start = self._capacity - 1
        self._end = self._size
        self._reversed = False
        self._data = new_data

    def get_at(self, index: int) -> Any | None:
        """
        Get element at the given index.
        Return None if index is out of bounds.
        Time complexity for full marks: O(1)
        """
        if self._size == 0:
            return None
        if index >= self._capacity or index < 0:
            return None
        rev = -1 if self._reversed else 1
        i = (self._start + (rev) * (1 + index)) % self._capacity
        element = self._data[i]
        return element

    def __getitem__(self, index: int) -> Any | None:
        """
        Same as get_at.
        Allows to use square brackets to index elements.
        """
        if self._size == 0:
            return None
        if index >= self._capacity or index < 0:
            return None
        rev = -1 if self._reversed else 1
        i = (self._start + (rev) * (1 + index)) % self._capacity
        element = self._data[i]
        return element
    
    def set_at(self, index: int, element: Any) -> None:
        """
        Get element at the given index.
        Do not modify the list if the index is out of bounds.
        Time complexity for full marks: O(1)
        """
        if index >= self._size or index < 0:
            return
        rev = -1 if self._reversed else 1
        i = (self._start + (rev) * (1 + index)) % self._capacity
        self._data[i] = element

    def __setitem__(self, index: int, element: Any) -> None:
        """
        Same as set_at.
        Allows to use square brackets to index elements.
        """
        if index >= self._size or index < 0:
            return
        rev = -1 if self._reversed else 1
        i = (self._start + (rev) * (1 + index)) % self._capacity
        self._data[i] = element

    def append(self, element: Any) -> None:
        """
        Add an element to the back of the array.
        Time complexity for full marks: O(1*) (* means amortized)
        """ 
        if self._size == self._capacity:
            self.__resize()
        self._data[self._end] = element 
        self.__increment_end() 
        self._size += 1

    def prepend(self, element: Any) -> None:
        """
        Add an element to the front of the array.
        Time complexity for full marks: O(1*)
        """
        if self._size == self._capacity:
            self.__resize() 
        self._data[self._start] = element
        self.__update_start()
        self._size += 1
    
    def __update_start(self) -> None:
        if self._reversed:
            self._start += 1
        else:
            self._start -= 1
        self._start %= self._capacity
            
    def __increment_end(self) -> None:
        if self._reversed:
            self._end -= 1
        else:
            self._end += 1
        self._end %= self._capacity
    
    def __decrement_end(self):
        if self._reversed:
            self._end += 1
        else:
            self._end -= 1
        self._end %= self._capacity

    def reverse(self) -> None:
        """
        Reverse the array.
        Time complexity for full marks: O(1)
        """
        self._start, self._end = self._end, self._start
        self._reversed = not self._reversed

    def remove(self, element: Any) -> None:
        """
        Remove the first occurrence of the element from the array.
        If there is no such element, leave the array unchanged.
        Time complexity for full marks: O(N)
        """
        found = False
        for i, x in enumerate(self):
            if x == element:
                found = True
            if found:
                self[i] = self[i + 1]
        if found: 
            self[self._size - 1] = None
            self.__decrement_end()
            self._size -= 1
            
    def remove_at(self, index: int) -> Any | None:
        """
        Remove the element at the given index from the array and return the removed element.
        If there is no such element, leave the array unchanged and return None.
        Time complexity for full marks: O(N)
        """
        if index >= self._size or index < 0:
            return None
        element = self[index]
        for i in range(index, self._size - 1):
            self[i] = self[i + 1]
        self[self._size - 1] = None
        self.__decrement_end()
        self._size -= 1
        return element

    def is_empty(self) -> bool:
        """
        Boolean helper to tell us if the structure is empty or not
        Time complexity for full marks: O(1)
        """
        return not self._size

    def is_full(self) -> bool:
        """
        Boolean helper to tell us if the structure is full or not
        Time complexity for full marks: O(1)
        """
        return self._size == self._capacity

    def get_size(self) -> int:
        """
        Return the number of elements in the list
        Time complexity for full marks: O(1)
        """
        return self._size

    def get_capacity(self) -> int:
        """
        Return the total capacity (the number of slots) of the list
        Time complexity for full marks: O(1)
        """
        return self._capacity

    def __mapping(self, index) -> int:
        # memory index returns array index
        return (index - self._start - 1) % self._capacity

    def sort(self) -> None:
        """
        Sort elements inside _data based on < comparisons.
        Time complexity for full marks: O(NlogN)
        """
        if self._size == 0:
            return
        if self._reversed:
            self.reverse()
        self.__quicksort((self._start + 1) % self._capacity, (self._end - 1) % self._capacity)
    
    def __quicksort(self, low, high):
        # low = actual start
        # high = actual end
        if self._data[high] is None or self._data[low] is None:
            return
        if self.__mapping(high) - self.__mapping(low) == 0:
            return
        if self.__mapping(high) - self.__mapping(low) <= 9: 
            self.__bubble_sort(low, high)
            return
        
        if self.__mapping(low) < self.__mapping(high):
            l, r = self.__partition(low, high)
            if l != low:
                self.__quicksort(low, l - 1)
            if r != high:
                self.__quicksort(r + 1, high)

    def __partition(self, low, high):
        # [less than, unknown, middle = pivot, unknown, greater than]
        # choose the middle element as pivot
        if low < high:
            mid = (low + high) // 2
        else:
            mid = (low - self._capacity + high) // 2
            mid = mid if mid > 0 else mid + self._capacity
        ml = mid
        mr = mid
        data = self._data
        pivot = data[mid]
        l = low
        r = high
        while self.__mapping(l) < self.__mapping(ml):
            if data[l] < pivot:
                l = 0 if l + 1 >= self._capacity else l + 1# % self._capacity
            elif data[l] == pivot:
                data[l], data[ml - 1] = data[ml - 1], data[l]
                ml = self._capacity - 1 if ml - 1 < 0 else ml - 1# % self._capacity
            elif data[l] > pivot:
                data[l], data[r] = data[r], data[l]
                r = self._capacity - 1 if r - 1 < 0 else r - 1# % self._capacity

        while self.__mapping(r) > self.__mapping(mr):
            if data[r] > pivot:
                r = self._capacity - 1 if r - 1 < 0 else r - 1# % self._capacity
            elif data[r] == pivot:
                data[r], data[mr + 1] = data[mr + 1], data[r]
                mr = 0 if mr + 1 >= self._capacity else mr + 1# % self._capacity
            elif data[r] < pivot:
                data[r], data[l] = data[l], data[r]
                l = 0 if (l + 1) >= self._capacity else (l + 1)# % self._capacity
        return l, r
    
    def __bubble_sort(self, low, high):
        data = self._data
        for _ in range(self.__mapping(low), self.__mapping(high)):
            l = low
            r = high
            while self.__mapping(l) < self.__mapping(r):
                if data[l] > data[(l + 1) % self._capacity]:
                    data[l], data[(l + 1) % self._capacity] = data[(l + 1) % self._capacity], data[l]
                l = (l + 1) % self._capacity
