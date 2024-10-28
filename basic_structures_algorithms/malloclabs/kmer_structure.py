"""
Skeleton for COMP3506/7505 A1, S2, 2024
The University of Queensland
Joel Mackenzie and Vladimir Morozov

MallocLabs K-mer Querying Structure
"""

from typing import Any

"""
You may wish to import your data structures to help you with some of the
problems. Or maybe not.
"""
from structures.bit_vector import BitVector
from structures.dynamic_array import DynamicArray
from structures.linked_list import DoublyLinkedList, Node


class KmerStore:
    """
    A data structure for maintaining and querying k-mers.
    You may add any additional functions or member variables
    as you see fit.
    At any moment, the structure is maintaining n distinct k-mers.

    You may use Python lists; this includes slicing, len,
    append. This is intended to help with returning the python list in the freq_geq() query,
    and to enable you to check the length of incoming list types without needing to iterate
    over them.
    """

    def __init__(self, k: int) -> None:
        self._k = k
        self._size = 0
        self._kmers = []
        # AA: 0, AC: 1, AG:, 2, AT: 3...
        self._comp_kmers = []

    def __str__(self) -> str:
        return str(self._kmers)

    def read(self, infile: str) -> None:
        """
        Given a path to an input file, break the sequences into
        k-mers and load them into your data structure.
        """
        file = open("infile", "r")

    def batch_insert(self, kmers: list[str]) -> None:
        """
        Given a list of m k-mers, insert the matching ones
        (including all duplicates).
        [V2: Correction]
        If the data structure contains n elements, and the input kmer list
        contains m elements, the targeted time complexity is:
        O(m log m) + O(n + m) amortized time (or better, of course!)
        """
        self.k_sort(kmers)
        for i in kmers:
            # self.increment_comp_kmer(i[-2:])
            self._kmers.append(i)
            self._size += 1
        self.k_sort(self._kmers)

    def batch_delete(self, kmers: list[str]) -> None:
        """
        Given a list of m k-mers, delete the matching ones
        (including all duplicates).
        [V2: Correction]
        If the data structure contains n elements, and the input kmer list
        contains m elements, the targeted time complexity is:
        O(m log m) + O(n + m) amortized time (or better, of course!)
        """
        self.k_sort(kmers)
        new_data = []
        i = 0
        j = 0
        self._comp_kmers = []
        while i < len(kmers) and j < len(self._kmers):
            if kmers[i] > self._kmers[j]:
                # self.increment_comp_kmer(j[-2:])
                new_data.append(self._kmers[j])
                j += 1
            elif kmers[i] < self._kmers[j]:
                i += 1
            else:
                j += 1
        
        while j < len(self._kmers):
            # self.increment_comp_kmer(j[-2:])
            new_data.append(self._kmers[j])
            j += 1

        self._kmers = new_data
        self._size = len(new_data)
                    
    def freq_geq(self, m: int) -> list[str]:
        """
        Given an integer m, return a list of k-mers that occur
        >= m times in your data structure.
        Time complexity for full marks: O(n)
        """
        tmp_list = []
        curr_count = 1
        curr_kmer = self._kmers[0]
        
        for i in range(1, self._size):
            if self._kmers[i] == curr_kmer:
                curr_count += 1
            else:
                if curr_count >= m:
                    tmp_list.append(curr_kmer)
                curr_kmer = self._kmers[i]
                curr_count = 1
        
        if curr_count >= m:
            tmp_list.append(curr_kmer)
        
        return tmp_list

    def count(self, kmer: str) -> int:
        """
        Given a k-mer, return the number of times it appears in
        your data structure.
        Time complexity for full marks: O(log n)
        """
        start = self.bin_search_first(0, self._size - 1, kmer)
        if start == -1:
            return 0  # k-mer not found
        
        end = self.bin_search_last(start, self._size - 1, kmer)
        return end - start + 1  

    def count_geq(self, kmer: str) -> int:
        """
        Given a k-mer, return the total number of k-mers that
        are lexicographically greater or equal.
        Time complexity for full marks: O(log n)
        """
        low, high = 0, len(self._kmers)
        while low < high:
            mid = (low + high) // 2
            # print(self._kmers[mid])
            if self._kmers[mid] >= kmer:
                high = mid
            else:
                low = mid + 1
        return len(self._kmers) - high

    def compatible(self, kmer: str) -> int:
        """
        Given a k-mer, return the total number of compatible
        k-mers. You will be using the two suffix characters
        of the input k-mer to compare against the first two
        characters of all other k-mers.
        Time complexity for full marks: O(1) :-)
        """
        return self._comp_kmers[kmer[-2:]]

    def increment_comp_kmer(self, kmer_suff: str):
        self._comp_kmers[kmer_suff] += 1

    def bin_search_first(self, low, high, x):
        if high >= low:
            mid = (high + low) // 2
            if self._kmers[mid] == x:
                if mid == 0 or self._kmers[mid - 1] < x:
                    return mid
                else:
                    return self.bin_search_first(low, mid - 1, x)
            elif self._kmers[mid] > x:
                return self.bin_search_first(low, mid - 1, x)
            else:
                return self.bin_search_first(mid + 1, high, x)
        return -1
    
    def bin_search_last(self, low, high, x):
        if high >= low:
            mid = (high + low) // 2
            if self._kmers[mid] == x:
                if mid == high or self._kmers[mid + 1] > x:
                    return mid
                else:
                    return self.bin_search_last(mid + 1, high, x)
            elif self._kmers[mid] > x:
                return self.bin_search_last(low, mid - 1, x)
            else:
                return self.bin_search_last(mid + 1, high, x)
        return -1

    def k_sort(self, arr) -> None:
        """
        Sort elements inside _data based on < comparisons.
        Time complexity for full marks: O(NlogN)
        """
        size = len(arr)
        self.quicksort(0, size - 1, arr)
        return size
    
    def quicksort(self, low, high, arr):
        if high - low == 0:
            return
        if high - low <= 9: 
            self.bubble_sort(low, high, arr)
            return
        
        if low < high:
            l, r = self.partition(low, high, arr)
            if l != low:
                self.quicksort(low, l - 1, arr)
            if r != high:
                self.quicksort(r + 1, high, arr)

    def partition(self, low, high, data):
        # [less than, unknown, middle = pivot, unknown, greater than]
        # choose the middle element as pivot
        mid = (low + high) // 2
        ml = mid
        mr = mid
        pivot = data[mid]
        l = low
        r = high
        while l < ml:
            if data[l] < pivot:
                l += 1
            elif data[l] == pivot:
                data[l], data[ml - 1] = data[ml - 1], data[l]
                ml -= 1
            elif data[l] > pivot:
                data[l], data[r] = data[r], data[l]
                r -= 1

        while r > mr:
            if data[r] > pivot:
                r -= 1
            elif data[r] == pivot:
                data[r], data[mr + 1] = data[mr + 1], data[r]
                mr += 1
            elif data[r] < pivot:
                data[r], data[l] = data[l], data[r]
                l += 1
        return l, r

    def bubble_sort(self, low, high, data):
        for _ in range(low, high):
            l = low
            r = high
            while l < r:
                if data[l] > data[l + 1]:
                    data[l], data[l + 1] = data[l + 1], data[l]
                l = l + 1
