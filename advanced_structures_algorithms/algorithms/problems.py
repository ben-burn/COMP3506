"""
Skeleton for COMP3506/7505 A2, S2, 2024
The University of Queensland
Joel Mackenzie and Vladimir Morozov

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

You may wish to import your data structures to help you with some of the
problems. Or maybe not. We did it for you just in case.
"""
from structures.entry import Entry, Compound, Offer
from structures.dynamic_array import DynamicArray, DumbArray
from structures.linked_list import DoublyLinkedList
from structures.bit_vector import BitVector
from structures.graph import Graph, LatticeGraph, Node
from structures.map import Map, Set
from structures.pqueue import PriorityQueue
from structures.bloom_filter import BloomFilter
from structures.util import Hashable

import math
# import time
# from .pathfinding import bfs_traversal_chain


class Stack:
    def __init__(self):
        self._data = [None] * 2
        self._size = 0
        self._capacity = 2

    def __resize(self):
        """
        Resize the stack by doubling its capacity.
        """
        new_cap = self._capacity * 2
        new_data = [None] * new_cap
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data
        self._capacity = new_cap

    def push(self, item):
        """
        Add an item to the top of the stack.
        """
        if self._size == self._capacity:
            self.__resize()
        self._data[self._size] = item
        self._size += 1

    def pop(self):
        """
        Remove and return the item from the top of the stack.
        """
        if self._size == 0:
            return None
        self._size -= 1
        return self._data[self._size]

    def is_empty(self):
        """
        Check if the stack is empty.
        """
        return self._size == 0


def maybe_maybe_maybe(database: list[str], query: list[str]) -> list[str]:
    """
    Task 3.1: Maybe Maybe Maybe

    @database@ is an array of k-mers in our database.
    @query@ is an array of k-mers we want to search for.

    Return a list of query k-mers that are *likely* to appear in the database.

    Limitations:
        "It works":
            @database@ contains up to 1000 elements;
            @query@ contains up to 1000 elements.

        "Exhaustive":
            @database@ contains up to 100'000 elements;
            @query@ contains up to 100'000 elements.

        "Welcome to COMP3506":
            @database@ contains up to 1'000'000 elements;
            @query@ contains up to 500'000 elements.

    Each test will run over three false positive rates. These rates are:
        fp_rate = 10%
        fp_rate = 5%
        fp_rate = 1%.

    You must pass each test in the given time limit and be under the given
    fp_rate to get the associated mark for that test.
    """
    answer = []

    # create a bloom filter to store every possible kmer
    database_len = len(database)
    kbf = BloomFilter(database_len)
    # t1 = time.time()
    kbf.insert_from_kmer_list(database)
    # t2 = time.time()
    # print("inster list: ", t2-t1)

    # For each kmer in the list check if the bloom filter contains the kmer
    # t1 = time.time()
    for kmer in query:
        # if it does add to answer
        if kbf.contains(kmer, True):
            answer.append(kmer)
    # t2 = time.time()
    # print("check contains: ", t2-t1)
    return answer


# def dora(graph: Graph, start: int, symbol_sequence: str,
#          ) -> tuple[BitVector, list[Entry]]:
#     """
#     Task 3.2: Dora and the Chin Bicken

#     @graph@ is the input graph G; G might be disconnected; each node contains
#     a single symbol in the node's data field.
#     @start@ is the integer identifier of the start vertex.
#     @symbol_sequence@ is the input sequence of symbols, L, with length n.
#     All symbols are guaranteed to be found in G.

#     Return a BitVector encoding symbol_sequence via a minimum redundancy code.
#     The BitVector should be read from index 0 upwards (so, the first symbol is
#     encoded from index 0). You also need to return your codebook as a
#     Python list of unique Entries. The Entry key should correspond to the
#     symbol, and the value should be a string. More information below.

#     Limitations:
#         "It works":
#             @graph@ has up to 1000 vertices and up to 1000 edges.
#             the alphabet consists of up to 26 characters.
#             @symbol_sequence@ has up to 1000 characters.

#         "Exhaustive":
#             @graph@ has up to 100'000 vertices and up to 100'000 edges.
#             the alphabet consists of up to 1000 characters.
#             @symbol_sequence@ has up to 100'000 characters.

#         "Welcome to COMP3506":
#             @graph@ has up to 1'000'000 vertices and up to 1'000'000 edges.
#             the alphabet consists of up to 300'000 characters.
#             @symbol_sequence@ has up to 1'000'000 characters.

#     """
#     coded_sequence = BitVector()

#     """
#     list of Entry objects, each entry has key=symbol, value=str. The str
#     value is just an ASCII representation of the bits used to encode the
#     given key. For example: x = Entry("c", "1101")
#     """
#     codebook = []  # list[Entry]

#     # DO THE THING
#     # huffman

#     # Explore all reachable rooms and collect all gene symbols, s, located at each node
#     symbols, symbol_size = dora_traversal(graph, start)

#     # Build a gene frequency table T, map each symbol to its total frequency in g
#     T = Map()

#     for index in range(symbol_size):
#         symbol = symbols[index]
#         freq = T.find(symbol)
#         if freq is None:
#             T.insert_kv(symbol, 1)
#         else:
#             T.insert_kv(symbol, freq + 1)

#     # print("symbols: ", symbols, " freq table: ", T._data)

#     # print("freq size: ", T.get_size(), " seq size: ", len(symbol_sequence))

#     # Build minimum redundancy code via Huffmans algorithm to create a codebook
#     # maps each s to codeword c

#     # Create a priority queue (lower frequency = higher priority)
#     min_heap = PriorityQueue()

#     visited = Set()

#     for symbol, freq in T:
#         new_node = HNode(freq, symbol)
#         min_heap.insert(freq, new_node)
#         visited.insert(symbol)

#     # Build a huffman tree
#     while min_heap.get_size() > 1:

#         # Remove two nodes from pq
#         left_node = min_heap.remove_min()
#         right_node = min_heap.remove_min()

#         # Create a new internal node with a frequency equal to the sum of both nodes
#         new_freq = left_node.get_frequency() + right_node.get_frequency()
#         new_node = HNode(new_freq)
#         new_node.set_left(left_node)
#         new_node.set_right(right_node)

#         # Insert the new internal node back into the priority queue
#         min_heap.insert(new_freq, new_node)

#     huffman_root = min_heap.remove_min()

#     c_d = Map()
#     # Assign codes to symbols, c_d is a map {k: symbol, v: code}
#     create_codebook_iterative(huffman_root, codebook, c_d)

#     # print(c_d._data)
#     # print([str(entry) for entry in codebook])
#     # assert c_d.get_size() == len(codebook)

#     for symb in symbol_sequence:
#         # Get the code from c_d
#         bit_string = c_d.find(symb)

#         # Append each bit to coded_sequence
#         if bit_string is not None:
#             for b in bit_string:
#                 coded_sequence.append(int(b))

#     # tuple[BitVector, list[Entry]]
#     return (coded_sequence, codebook)


# def create_codebook(node, current_code: str, codebook: list, c_d: Map):
#     """
#     Recursively traverse the Huffman tree to create the codebook.
#     """
#     # Base case: leaf node (a symbol)
#     if node.get_left() is None and node.get_right() is None:
#         # Leaf node - it's a symbol, add it to the codebook and map
#         codebook.append(Entry(node.get_data(), current_code))  # Add to codebook list
#         c_d.insert_kv(node.get_data(), current_code)  # Add to map for fast lookup
#         return

#     # Recursive case: traverse left and right children
#     if node.get_left():
#         create_codebook(node.get_left(), current_code + "0", codebook, c_d)
#     if node.get_right():
#         create_codebook(node.get_right(), current_code + "1", codebook, c_d)


# def create_codebook_iterative(node, codebook: list, c_d: Map):
#     """
#     Iteratively create codebook from Huffman tree.
#     """
#     if node is None:
#         return

#     stack = Stack()
#     stack.push((node, ""))

#     # Iterative depth-first traversal
#     while not stack.is_empty():
#         node, current_code = stack.pop()

#         # If leaf node, store the code for this symbol
#         if node.get_left() is None and node.get_right() is None:
#             # Add the symbol and its code to the codebook and map
#             codebook.append(Entry(node.get_data(), current_code))
#             c_d.insert_kv(node.get_data(), current_code)

#         # If right child exists, push it onto the stack with updated code (append "1")
#         if node.get_right():
#             stack.push((node.get_right(), current_code + "1"))

#         # If left child exists, push it onto the stack with updated code (append "0")
#         if node.get_left():
#             stack.push((node.get_left(), current_code + "0"))


# class HNode():
#     def __init__(self, frequency, data=None):
#         self._left = None
#         self._right = None
#         self._frequency = frequency
#         self._data = data

#     def __str__(self):
#         return f"{self._data}"

#     def set_left(self, h_node):
#         self._left = h_node

#     def set_right(self, h_node):
#         self._right = h_node

#     def get_left(self):
#         return self._left

#     def get_right(self):
#         return self._right

#     def get_data(self):
#         return self._data

#     def get_frequency(self) -> int:
#         return self._frequency

#     def __lt__(self, other):
#         return self._frequency < other.get_frequency()

#     def __eq__(self, other):
#         return self._frequency == other.get_frequency()


# def dora_traversal(graph, start):

#     # Priority Queue - stores nodes
#     node_num = len(graph._nodes)
#     q = [None] * node_num
#     q_ptr = 0
#     end_ptr = 1
#     q[q_ptr] = start

#     # Set to keep track of visited nodes
#     visited = Set()
#     visited.insert(start)

#     # Symbols to return
#     # symbols = [None] * node_num
#     symbols = DynamicArray()

#     # symbols[0] = get_symbol(graph, start)
#     # s_ptr = 1
#     symbols.append(get_symbol(graph, start))

#     while end_ptr > q_ptr:
#         # Ger curr node
#         curr_node_id = q[q_ptr]
#         q_ptr += 1

#         # Explore neighbors
#         for neighbor_info in graph.get_neighbours(curr_node_id):
#             if isinstance(neighbor_info, tuple):
#                 # If it's a weighted graph, neighbor_info is a (neighbor, weight) tuple
#                 neighbor, weight = neighbor_info
#             else:
#                 # If it's an unweighted graph, neighbor_info is just the neighbor node
#                 neighbor = neighbor_info

#             node_id = neighbor.get_id()

#             # If node hasn't been visited yet, process it
#             if visited.find(node_id) is None:
#                 visited.insert(node_id)
#                 q[end_ptr] = node_id
#                 end_ptr += 1
#                 # symbols[s_ptr] = get_symbol(graph, node_id)
#                 # s_ptr += 1
#                 symbols.append(get_symbol(graph, node_id))

#     return symbols, symbols.get_size()


# def get_symbol(graph, int):
#     return graph.get_node(int).get_data()


def dora(graph: Graph, start: int, symbol_sequence: str,
         ) -> tuple[BitVector, list[Entry]]:
    """
    Task 3.2: Dora and the Chin Bicken

    @graph@ is the input graph G; G might be disconnected; each node contains
    a single symbol in the node's data field.
    @start@ is the integer identifier of the start vertex.
    @symbol_sequence@ is the input sequence of symbols, L, with length n.
    All symbols are guaranteed to be found in G.

    Return a BitVector encoding symbol_sequence via a minimum redundancy code.
    The BitVector should be read from index 0 upwards (so, the first symbol is
    encoded from index 0). You also need to return your codebook as a
    Python list of unique Entries. The Entry key should correspond to the
    symbol, and the value should be a string. More information below.

    Limitations:
        "It works":
            @graph@ has up to 1000 vertices and up to 1000 edges.
            the alphabet consists of up to 26 characters.
            @symbol_sequence@ has up to 1000 characters.

        "Exhaustive":
            @graph@ has up to 100'000 vertices and up to 100'000 edges.
            the alphabet consists of up to 1000 characters.
            @symbol_sequence@ has up to 100'000 characters.

        "Welcome to COMP3506":
            @graph@ has up to 1'000'000 vertices and up to 1'000'000 edges.
            the alphabet consists of up to 300'000 characters.
            @symbol_sequence@ has up to 1'000'000 characters.

    """
    coded_sequence = BitVector()

    """
    list of Entry objects, each entry has key=symbol, value=str. The str
    value is just an ASCII representation of the bits used to encode the
    given key. For example: x = Entry("c", "1101")
    """
    codebook = []  # list[Entry]

    # DO THE THING
    # huffman

    # Explore all reachable rooms and collect all gene symbols, s, located at each node
    # t1 = time.time()
    symbols, symbol_size = dora_traversal(graph, start)
    # print(symbols, symbol_size)
    # t2 = time.time()
    # print("bfs: ", t2-t1)

    # Build a gene frequency table T, map each symbol to its total frequency in g
    T = Map()
    for index in range(symbol_size):
        symbol = symbols[index]
        freq = T.find(symbol)
        if freq is None:
            T.insert_kv(symbol, 1)
        else:
            T.insert_kv(symbol, freq + 1)

    # Build minimum redundancy code via Huffmans algorithm to create a codebook
    # maps each s to codeword c

    # Create a priority queue (lower frequency = higher priority)
    min_heap = PriorityQueue()
    # entries = T.to_list()
    entries = T._data
    for entry in entries:
        if entry is not None:
            if entry.get_value() is not None:
                symbol = entry.get_key()
                freq = entry.get_value()
                new_node = HNode(freq, symbol)
                min_heap.insert(freq, new_node)

    # Build a huffman tree
    while min_heap.get_size() > 1:
        # Remove two nodes from pq
        left_freq, left_node = min_heap.remove_min_with_key()
        right_freq, right_node = min_heap.remove_min_with_key()

        # Create new node which has priority as the sum of the two nodes
        new_symbol = f'({left_node.get_data()}+{right_node.get_data()})'
        new_freq = left_freq + right_freq
        new_node = HNode(new_freq, new_symbol)
        new_node.set_left(left_node)
        new_node.set_right(right_node)

        # Insert back into pq
        min_heap.insert(new_freq, new_node)

    c_d = Map()
    # Assign codes to symbols
    create_codebook(min_heap.remove_min(), "", codebook, c_d)

    # print("codebook: ")
    for symb in symbol_sequence:
        bit_string = c_d.find(symb)
        if bit_string is not None:
            for b in bit_string:
                coded_sequence.append(int(b))

    # tuple[BitVector, list[Entry]]
    return (coded_sequence, codebook)


def create_codebook(node, current_code, codebook, c_d: Map):
    """
    Recursively traverse the Huffman tree to create the codebook.
    """
    if node.get_left() is None and node.get_right() is None:
        # Leaf node - it's a symbol, add it to the codebook
        c_d.insert_kv(node.get_data(), current_code)
        codebook.append(Entry(node.get_data(), current_code))
        return

    # Traverse left and right children, appending '0' for left and '1' for right
    if node.get_left():
        create_codebook(node.get_left(), current_code + "0", codebook, c_d)
    if node.get_right():
        create_codebook(node.get_right(), current_code + "1", codebook, c_d)


class HNode():
    def __init__(self, frequency, data=None):
        self._left = None
        self._right = None
        self._frequency = frequency
        self._data = data

    def set_left(self, h_node):
        self._left = h_node

    def set_right(self, h_node):
        self._right = h_node

    def get_left(self):
        return self._left

    def get_right(self):
        return self._right

    def get_data(self):
        return self._data

    def get_frequency(self) -> int:
        return self._frequency

    def __lt__(self, other):
        return self._frequency < other.get_frequency()

    def __eq__(self, other):
        return self._frequency == other.get_frequency()


def dora_traversal(graph, start):

    # Priority Queue - stores nodes
    node_num = len(graph._nodes)
    q = [None] * node_num
    q_ptr = 0
    end_ptr = 1
    q[q_ptr] = start

    # Set to keep track of visited nodes
    visited = Set()
    visited.insert(start)

    # Symbols to return
    symbols = [None] * node_num

    symbols[0] = get_symbol(graph, start)
    s_ptr = 1

    while end_ptr > q_ptr:
        # Ger curr node
        curr_node_id = q[q_ptr]
        q_ptr += 1

        # Explore neighbors
        for node in graph.get_neighbours(curr_node_id):
            node_id = node.get_id()

            # If node hasn't been visited yet, process it
            if visited.find(node_id) is None:
                visited.insert(node_id)
                q[end_ptr] = node_id
                end_ptr += 1
                symbols[s_ptr] = get_symbol(graph, node_id)
                s_ptr += 1

    return symbols, s_ptr


def get_symbol(graph, int):
    return graph.get_node(int).get_data()


def chain_reaction(compounds: list[Compound]) -> int:
    """
    Task 3.3: Chain Reaction

    @compounds@ is a list of Compound types, see structures/entry.py for the
    definition of a Compound. In short, a Compound has an integer x and y
    coordinate, a floating point radius, and a unique integer representing
    the compound identifier.

    Return the compound identifier of the compound that will yield the
    maximal number of compounds in the chain reaction if set off. If there
    are ties, return the one with the smallest identifier.

    Limitations:
        "It works":
            @compounds@ has up to 100 elements

        "Exhaustive":
            @compounds@ has up to 1000 elements

        "Welcome to COMP3506":
            @compounds@ has up to 10'000 elements

    """
    maximal_compound = -1
    longest_reaction = 0

    node_num = len(compounds)

    adj_list = [None] * node_num
    for i in range(node_num):
        adj_list[i] = DynamicArray()

    # Create adjacency list based off compounds radius
    for i, comp_i in enumerate(compounds):
        for j, comp_j in enumerate(compounds):
            if i != j:
                # If comp_j is within the radius of comp_i
                if check_radius(comp_i, comp_j):
                    adj_list[i].append(j)

    for i in range(node_num):
        reaction_len = bfs_chain(i, node_num, adj_list)
        # Check if this reaction is longer, or if it's a tie but has a smaller compound ID
        if reaction_len > longest_reaction or (reaction_len == longest_reaction and
                                               compounds[i].get_compound_id() < maximal_compound):
            longest_reaction = reaction_len
            maximal_compound = compounds[i].get_compound_id()

    return maximal_compound


def check_radius(compound1: Compound, compound2: Compound) -> bool:
    x1, y1 = compound1.get_coordinates()
    x2, y2 = compound2.get_coordinates()
    r1 = compound1.get_radius()

    # Calculate Euclidean distance between the two compounds
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance <= r1


def bfs_chain(start: int, node_num: int, adj_list: list[DynamicArray]) -> int:
    visited = [False] * node_num
    pq = PriorityQueue()
    pq.insert_fifo(start)
    visited[start] = True
    chain_length = 1  # Count the starting compound
    while not pq.is_empty():
        node = pq.remove_min()
        ns = adj_list[node]
        ns_size = ns.get_size()
        for i in range(ns_size):
            neighbour = ns[i]
            if not visited[neighbour]:
                visited[neighbour] = True
                chain_length += 1
                pq.insert_fifo(neighbour)

    return chain_length


def labyrinth(offers: list[Offer]) -> tuple[int, int]:
    """
    Task 3.4: Labyrinth

    @offers@ is a list of Offer types, see structures/entry.py for the
    definition of an Offer. In short, an Offer stores n (number of nodes),
    m (number of edges), and k (diameter) of the given Labyrinth. Each
    Offer also has an associated cost, and a unique offer identifier.

    Return the offer identifier and the associated cost for the cheapest
    labyrinth that can be constructed from the list of offers. If there
    are ties, return the one with the smallest identifier.
    You are guaranteed that all offer ids are distinct.

    Limitations:
        "It works":
            @offers@ contains up to 1000 items.
            0 <= n <= 1000
            0 <= m <= 1000
            0 <= k <= 1000

        "Exhaustive":
            @offers@ contains up to 100'000 items.
            0 <= n <= 10^6
            0 <= m <= 10^6
            0 <= k <= 10^6

        "Welcome to COMP3506":
            @offers@ contains up to 5'000'000 items.
            0 <= n <= 10^42
            0 <= m <= 10^42
            0 <= k <= 10^42

    """
    best_offer_id = -1
    best_offer_cost = float('inf')

    # DO THE THING
    # For each offer in list
    for offer in offers:
        # check connection and edges
        if check_connection(offer):
            # check the shortest simple path between any two vertices is at most k
            if check_k(offer):
                # if pass all these checks, check cost
                if offer.get_cost() < best_offer_cost:
                    # update best_offer_id and best_offer_cost
                    best_offer_id = offer.get_offer_id()
                    best_offer_cost = offer.get_cost()

    return (best_offer_id, best_offer_cost)


def check_connection(offer: Offer):
    nodes = offer.get_n()
    edges = offer.get_m()

    # Check zero
    if not check_zero(offer):
        return False

    # Check minimum number of edges to be connected
    if edges < nodes - 1:
        return False

    # Check maximum number of edges without double edges
    if edges > max_edges(nodes):
        return False

    return True


def check_k(offer: Offer):
    nodes = offer.get_n()
    edges = offer.get_m()
    k = offer.get_k()

    # Every node must connect to every other node
    if k == 1 and edges != max_edges(nodes):
        return False
    elif not (edges > nodes - 1 and edges <= max_edges(nodes)):
        return False

    return True


def max_edges(n: int):
    return (n * (n - 1)) / 2


def check_zero(offer: Offer):
    nodes = offer.get_n()
    edges = offer.get_m()
    k = offer.get_k()

    if nodes == 0:
        # Edges must be 0
        return not edges

    if k == 0:
        # Nodes > 0
        return (nodes == 1 and edges == 0)

    # k > 0 and nodes > 0
    return edges
