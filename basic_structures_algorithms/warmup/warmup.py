"""
Skeleton for COMP3506/7505 A1, S2, 2024
The University of Queensland
Joel Mackenzie and Vladimir Morozov

WARMUP PROBLEMS

 Each problem will be assessed on three sets of tests:

1. "It works":
       Basic inputs and outputs, including the ones peovided as examples, with generous time and memory restrictions.
       Large inputs will not be tested here.
       The most straightforward approach will likely fit into these restrictions.

2. "Exhaustive":
       Extensive testing on a wide range of inputs and outputs with tight time and memory restrictions.
       These tests won't accept brute force solutions, you'll have to apply some algorithms and optimisations.

 3. "Welcome to COMP3506":
       Extensive testing with the tightest possible time and memory restrictions
       leaving no room for redundant operations.
       Every possible corner case will be assessed here as well.

There will be hidden tests in each category that will be published only after the assignment deadline.
"""

"""
You may wish to import your data structures to help you with some of the
problems. Or maybe not. We did it for you just in case.
"""
from structures.bit_vector import BitVector
from structures.dynamic_array import DynamicArray
from structures.linked_list import DoublyLinkedList, Node


def main_character(instring: list[int]) -> int:
    """
    @instring@ is an array of integers in the range [0, 2^{32}-1].
    Return the first position a repeat integer is encountered, or -1 if
    there are no repeated ints.

    Limitations:
        "It works":
            @instring@ may contain up to 10'000 elements.

        "Exhaustive":
            @instring@ may contain up to 300'000 elements.

        "Welcome to COMP3506":
            @instring@ may contain up to 5'000'000 elements.

    Examples:
    main_character([1, 2, 3, 4, 5]) == -1
    main_character([1, 2, 1, 4, 4, 4]) == 2
    main_character([7, 1, 2, 7]) == 3
    main_character([60000, 120000, 654321, 999, 1337, 133731337]) == -1
    """
    if len(instring) == 0:
        return -1
    
    bv = DynamicArray()
    BITS_PER_ELEMENT = 32
    bv.bv_init(BITS_PER_ELEMENT)
    for index, x in enumerate(instring):
        arr_ind = int(x // BITS_PER_ELEMENT)
        position = int(x % BITS_PER_ELEMENT)
        num = bv[arr_ind]
        if ((num >> position) & 1) == 1: 
            return index
        else:
            bv[arr_ind] = bv[arr_ind] | 1 << position
    return -1

def missing_odds(inputs: list[int]) -> int:
    """
    @inputs@ is an unordered array of distinct integers.
    If @a@ is the smallest number in the array and @b@ is the biggest,
    return the sum of odd numbers in the interval [a, b] that are not present in @inputs@.
    If there are no such numbers, return 0.

    Limitations:
        "It works":
            @inputs@ may contain up to 10'000 elements.
            Each element is in range 0 <= inputs[i] <= 10^4
        "Exhaustive":
            @inputs@ may contain up to 300'000 elements.
            Each element is in range 0 <= inputs[i] <= 10^6
        "Welcome to COMP3506":
            @inputs@ may contain up to 5'000'000 elements.
            Each element is in range 0 <= inputs[i] <= 10^16

    Examples:
    missing_odds([1, 2]) == 0
    missing_odds([1, 3]) == 0
    missing_odds([1, 4]) == 3
    missing_odds([4, 1]) == 3
    missing_odds([4, 1, 8, 5]) == 10    # 3 and 7 are missing
    """
    if len(inputs) == 0:
        return 0
    # dy = DynamicArray()
    min = inputs[0]
    max = inputs[0]
    rem_sum = 0
    for x in inputs:
        # print(min, max, rem_sum)
        if x < min:
            min = x
        if x > max:
            max = x
        if x % 2 == 1:
            rem_sum += x
    
    first = min if min % 2 else min + 1
    last = max if max % 2 else max - 1
    # n = number of odd numbers between min and max (inclusive)
    n = (last - first) // 2 + 1 if first <= last else 0
    ret_sum = (n * (first + last)) // 2

    return ret_sum - rem_sum

# def old_k_cool(k: int, n: int) -> int:
#     MODULUS = 10**16 + 61

#     answer = 0
#     count = 0
#     while n:
#         if n % 2:
#             answer += k**count
#         count += 1
#         n = n >> 1

#     return answer % MODULUS

def k_cool(k: int, n: int) -> int:
    """
    Return the n-th smallest k-cool number for the given @n@ and @k@.
    The result can be large, so return the remainder of division of the result
    by 10^16 + 61 (this constant is provided).

    Limitations:
        "It works":
            2 <= k <= 128
            1 <= n <= 10000
        "Exhaustive":
            2 <= k <= 10^16
            1 <= n <= 10^100     (yes, that's ten to the power of one hundred)
        "Welcome to COMP3506":
            2 <= k <= 10^42
            1 <= n <= 10^100000  (yes, that's ten to the power of one hundred thousand)

    Examples:
    k_cool(2, 1) == 1                     # The first 2-cool number is 2^0 = 1
    k_cool(2, 3) == 2                     # The third 2-cool number is 2^1 + 2^0 = 3
    k_cool(3, 5) == 10                    # The fifth 3-cool number is 3^2 + 3^0 = 10
    k_cool(10, 42) == 101010
    k_cool(128, 5000) == 9826529652304384 # The actual result is larger than 10^16 + 61,
                                          # so k_cool returns the remainder of division by 10^16 + 61
    """

    MODULUS = 10**16 + 61
    answer = 0

    # Reverse bits
    dy = DynamicArray()
    while n:
        dy.append(n & 1)
        n >>= 1
    dy.reverse()

    for x in dy:
        answer *= k
        if x == 1:
            answer += 1
        answer %= MODULUS
    
    return answer

def number_game(numbers: list[int]) -> tuple[str, int]:
    """
    @numbers@ is an unordered array of integers. The array is guaranteed to be of even length.
    Return a tuple consisting of the winner's name and the winner's score assuming that both play optimally.
    "Optimally" means that each player makes moves that maximise their chance of winning
    and minimise opponent's chance of winning.
    You are ALLOWED to use a tuple in your return here, like: return (x, y)
    Possible string values are "Alice", "Bob", and "Tie"

    Limitations:
        "It works":
            @numbers@ may contain up to 10'000 elements.
            Each element is in range 0 <= numbers[i] <= 10^6
        "Exhaustive":
            @numbers@ may contain up to 100'000 elements.
            Each element is in range 0 <= numbers[i] <= 10^16
        "Welcome to COMP3506":
            @numbers@ may contain up to 300'000 elements.
            Each element is in range 0 <= numbers[i] <= 10^16

    Examples:
    number_game([5, 2, 7, 3]) == ("Bob", 5)
    number_game([3, 2, 1, 0]) == ("Tie", 0)
    number_game([2, 2, 2, 2]) == ("Alice", 4)

    For the second example, if Alice picks 2 to increase her score, Bob will pick 3 and win. Alice does not want that.
    The same happens if she picks 1 or 0, but this time she won't even increase her score.
    The only scenario when Bob does not win immediately is if Alice picks 3.
    Then, Bob faces the same choice:
    pick 1 to increase his score knowing that Alice will pick 2 and win, or pick 2 himself.
    The same happens on the next move.
    So, nobody picks any numbers to increase their score, which results in a Tie with both players having scores of 0.
    """
    if len(numbers) == 0:
        return ("Tie", 0)
    # Sort
    alice = ("Alice", 0)
    bob = ("Bob", 0)
    size = my_sort(numbers)
    i = size - 1
    while i >= 0:
        if ((i + 1) % 2 == 0):
            if numbers[i] % 2 == 0:
                alice = (alice[0], alice[1] + numbers[i])
        else:
            if numbers[i] % 2:
                bob = (bob[0], bob[1] + numbers[i])
        i -= 1
    if alice[1] == bob[1]: return ("Tie", 0)
    return alice if alice[1] > bob[1] else bob

def road_illumination(road_length: int, poles: list[int]) -> float:
    """
    @poles@ is an unordered array of integers.
    Return a single floating point number representing the smallest possible radius of illumination
    required to illuminate the whole road.
    Floating point numbers have limited precision. Your answer will be accepted
    if the relative or absolute error does not exceed 10^(-6),
    i.e. |your_ans - true_ans| <= 0.000001 OR |your_ans - true_ans|/true_ans <= 0.000001

    Limitations:
        "It works":
            @poles@ may contain up to 10'000 elements.
            0 <= @road_length@ <= 10^6
            Each element is in range 0 <= poles[i] <= 10^6
        "Exhaustive":
            @poles@ may contain up to 100'000 elements.
            0 <= @road_length@ <= 10^16
            Each element is in range 0 <= poles[i] <= 10^16
        "Welcome to COMP3506":
            @poles@ may contain up to 300'000 elements.
            0 <= @road_length@ <= 10^16
            Each element is in range 0 <= poles[i] <= 10^16

    Examples:
    road_illumination(15, [15, 5, 3, 7, 9, 14, 0]) == 2.5
    road_illumination(5, [2, 5]) == 2.0
    """
    if road_length == 0 or len(poles) == 0:
        return 0
    # Max distance between poles
    max_d = 0
    # Sorts array and returns size of array
    size = my_sort(poles)
    # For each 2 elements check the if the distance between is 
    # greater than max size
    for i in range(size - 1):
        diff = poles[i + 1] - poles[i]
        max_d = diff if diff > max_d else max_d
    # The max radius
    max_r = max_d / 2

    max_r = max_r if poles[0] < max_r else poles[0]
    max_r = max_r if road_length - poles[size - 1] < max_r else road_length - poles[size - 1]
    return max_r

def my_sort(arr) -> None:
        """
        Sort elements inside _data based on < comparisons.
        Time complexity for full marks: O(NlogN)
        """
        size = len(arr)
        quicksort(0, size - 1, arr)
        return size
    
def quicksort(low, high, arr):
    if high - low == 0:
        return
    if high - low <= 9: 
        bubble_sort(low, high, arr)
        return
    
    if low < high:
        l, r = partition(low, high, arr)
        if l != low:
            quicksort(low, l - 1, arr)
        if r != high:
            quicksort(r + 1, high, arr)

def partition(low, high, data):
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

def bubble_sort(low, high, data):
    for _ in range(low, high):
        l = low
        r = high
        while l < r:
            if data[l] > data[l + 1]:
                data[l], data[l + 1] = data[l + 1], data[l]
            l = l + 1