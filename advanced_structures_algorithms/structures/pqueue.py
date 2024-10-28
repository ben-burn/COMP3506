"""
Skeleton for COMP3506/7505 A2, S2, 2024
The University of Queensland
Joel Mackenzie and Vladimir Morozov
"""

from typing import Any
from structures.entry import Entry
from structures.dynamic_array import DynamicArray

# import math


class PriorityQueue:
    """
    An implementation of the PriorityQueue ADT. We have used the implicit
    tree method: an array stores the data, and we use the heap shape property
    to directly index children/parents.

    The provided methods consume keys and values. Keys are called "priorities"
    and should be comparable numeric values; smaller numbers have higher
    priorities.
    Values are called "data" and store the payload data of interest.
    We use the Entry types to store (k, v) pairs.
    """

    def __init__(self):
        """
        Empty construction
        """
        self._arr = DynamicArray()
        self._max_priority = 0

    def __str__(self) -> str:
        return str(self._arr)

    def _parent(self, ix: int) -> int:
        """
        Given index ix, return the index of the parent
        """
        if ix == 0:
            return ix
        return (ix - 1) // 2

    def insert(self, priority: int, data: Any) -> None:
        """
        Insert some data to the queue with a given priority.
        """
        new = Entry(priority, data)
        # Put it at the back of the heap
        self._arr.append(new)
        ix = self._arr.get_size() - 1
        # Now swap it upwards with its parent until heap order is restored
        while ix > 0 and self._arr[ix].get_key() < self._arr[self._parent(ix)].get_key():
            parent_ix = self._parent(ix)
            self._arr[ix], self._arr[parent_ix] = self._arr[parent_ix], self._arr[ix]
            ix = parent_ix

    def insert_fifo(self, data: Any) -> None:
        """
        Insert some data to the queue in FIFO mode. Note that a user
        should never mix `insert` and `insert_fifo` calls, and we assume
        that nobody is silly enough to do this (we do not test this).
        """
        self.insert(self._max_priority, data)
        self._max_priority += 1

    def get_min_priority(self) -> Any:
        """
        Return the priority of the min element
        """
        if self.is_empty():
            return None
        return self._arr[0].get_key()

    def get_min_value(self) -> Any:
        """
        Return the highest priority value from the queue, but do not remove it
        """
        if self.is_empty():
            return None
        return self._arr[0].get_value()

    def remove_min(self) -> Any:
        """
        Extract (remove) the highest priority value from the queue.
        You must then maintain the queue to ensure priority order.
        """
        return self.__remove_helper().get_value()

    def remove_min_with_key(self) -> tuple[Any, Any]:
        entry = self.__remove_helper()
        key = entry.get_key()
        value = entry.get_value()
        return (key, value)

    def __remove_helper(self) -> Entry:
        if self.is_empty():
            return None

        result = self._arr[0]

        last = self.get_size() - 1
        self._arr[0] = self._arr[last]
        self._arr.remove_at(last)

        cur = 0
        size = self.get_size()
        while cur < size:
            left = 2 * cur + 1
            right = 2 * cur + 2

            smallest = cur
            if left < size and self._arr[smallest].get_key() > self._arr[left].get_key():
                smallest = left
            if right < size and self._arr[smallest].get_key() > self._arr[right].get_key():
                smallest = right

            # if key is same compare value
            if left < size and self._arr[smallest].get_key() == self._arr[left].get_key():
                if self._arr[smallest].get_value() > self._arr[left].get_value():
                    smallest = left

            if smallest != cur:
                self._arr[cur], self._arr[smallest] = self._arr[smallest], self._arr[cur]
                cur = smallest
            else:
                break

        return result

    def get_size(self) -> int:
        """
        Does what it says on the tin
        """
        return self._arr.get_size()

    def is_empty(self) -> bool:
        """
        Ditto above
        """
        return self._arr.is_empty()

    def ip_build(self, input_list: DynamicArray) -> None:
        """
        Take ownership of the list of Entry types, and build a heap
        in-place. That is, turn input_list into a heap, and store it
        inside the self._arr as a DynamicArray. You might like to
        use the DynamicArray build_from_list function. You must use
        only O(1) extra space.
        """
        if isinstance(input_list, list):
            self._arr.build_from_list(input_list)
        elif isinstance(input_list, DynamicArray):
            self._arr = input_list
        length = self._arr.get_size()
        if not length:
            return
        # height = math.floor(math.log2(length))
        # leaf_num = 2**height
        for index in range((length // 2) - 1, -1, -1):
            self.__heap_helper(length, index)

    def __heap_helper(self, length, index):
        """
        Recursively heapifies a tree
        """
        # parent index (currently assumed smallest)
        smallest = index

        # left child index
        left = 2 * index + 1

        # right child index
        right = 2 * index + 2

        # Determine if smallest element is left child
        if left < length and self._arr[left] < self._arr[smallest]:
            smallest = left

        # Determine if smallest element is right child
        if right < length and self._arr[right] < self._arr[smallest]:
            smallest = right

        # If left or right child are smaller than parent, swap then recursively check subtree
        if smallest != index:
            self._arr[index], self._arr[smallest] = self._arr[smallest], self._arr[index]
            self.__heap_helper(length, smallest)

    def sort(self) -> DynamicArray:
        """
        Use HEAPSORT to sort the heap being maintained in self._arr, using
        self._arr to store the output (in-place). You must use only O(1)
        extra space. Once sorted, return self._arr (the DynamicArray of
        Entry types).

        Once this sort function is called, the heap can be considered as
        destroyed and will not be used again (hence returning the underlying
        array back to the caller).
        """
        length = self._arr.get_size()
        # Swap the largest element with the root, heapify sub-trees
        for index in range((length - 1), 0, -1):
            self._arr[index], self._arr[0] = self._arr[0], self._arr[index]
            self.__heap_helper(index, 0)

        # Reverse the list so that it is in ascending order
        # for i in range(length // 2):
        #     self._arr[i], self._arr[length - i -
        #                             1] = self._arr[length - i - 1], self._arr[i]

        ret_arr = self._arr
        self._arr = DynamicArray()
        return ret_arr

    def update(self, old_key: Any, new_key: Any, id: Any):
        for entry in self._arr:
            if entry is None:
                break
            if entry.get_key() == old_key and entry.get_value() == id:
                entry.update_key(new_key)

        length = self._arr.get_size()
        for index in range((length // 2) - 1, -1, -1):
            self.__heap_helper(length, index)
