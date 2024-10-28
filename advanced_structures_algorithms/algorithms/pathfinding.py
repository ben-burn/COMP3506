"""
Skeleton for COMP3506/7505 A2, S2, 2024
The University of Queensland
Joel Mackenzie and Vladimir Morozov

You may wish to import your data structures to help you with some of the
problems. Or maybe not. We did it for you just in case.
"""
from structures.entry import Entry
from structures.dynamic_array import DynamicArray
from structures.graph import Graph, LatticeGraph
from structures.map import Map
from structures.pqueue import PriorityQueue
# from structures.bloom_filter import BloomFilter
from structures.util import Hashable


def bfs_traversal(
    graph: Graph | LatticeGraph, origin: int, goal: int
) -> tuple[DynamicArray, DynamicArray]:
    """
    Task 2.1: Breadth First Search

    @param: graph
      The general graph or lattice graph to process
    @param: origin
      The ID of the node from which to start traversal
    @param: goal
      The ID of the target node

    @returns: tuple[DynamicArray, DynamicArray]
      1. The ordered path between the origin and the goal in node IDs
      (or an empty DynamicArray if no path exists);
      2. The IDs of all nodes in the order they were visited.
    """

    # Stores the keys of the nodes in the order they were visited
    visited_order = DynamicArray()

    # Stores the path from the origin to the goal
    path = DynamicArray()

    # Priority Queue - stores nodes
    # Priority Queue - stores nodes
    node_num = len(graph._nodes)
    q = [None] * node_num
    q_ptr = 0
    end_ptr = 1
    q[q_ptr] = origin

    # Stores Key: nodeID, Value: Previous nodeID
    chain = Map()
    chain.insert_kv(origin, -1)

    while end_ptr > q_ptr:
        curr_node_id = q[q_ptr]
        q_ptr += 1

        visited_order.append(curr_node_id)

        # Check if goal is reached
        if curr_node_id == goal:
            bfs_end(chain, goal, path)
            return (path, visited_order)

        # Explore neighbors
        for node in graph.get_neighbours(curr_node_id):
            node_id = node.get_id()

            # If node hasn't been visited yet, process it
            if chain.find(node_id) is None:
                # Mark the node as visited by setting the predecessor
                chain.insert_kv(node_id, curr_node_id)

                q[end_ptr] = node_id
                end_ptr += 1

    return (path, visited_order)


def bfs_end(chain, goal, path):
    temp = DynamicArray()
    temp.append(goal)
    node_id = goal
    while chain.find(node_id) != -1:
        # print(node_id)
        node_id = chain.find(node_id)
        temp.append(node_id)

    path_len = temp.get_size()
    path.allocate(path_len, None)

    for i in range(path_len - 1, -1, -1):
        path[path_len - 1 - i] = temp[i]

    return True


def dijkstra_traversal(graph: Graph, origin: int) -> DynamicArray:
    """
    Task 2.2: Dijkstra Traversal

    @param: graph
      The *weighted* graph to process (POSW graphs)
    @param: origin
      The ID of the node from which to start traversal.

    @returns: DynamicArray containing Entry types.
      The Entry key is a node identifier, Entry value is the cost of the
      shortest path to this node from the origin.

    NOTE: Dijkstra does not work (by default) on LatticeGraph types.
    This is because there is no inherent weight on an edge of these
    graphs. It should of course work where edge weights are uniform.
    """
    # ALGO GOES HERE
    #  py test_pathfinding.py --dijkstra --graph POSW/test/simple1.graph --seed 2
    valid_locations = DynamicArray()
    weights = Map()
    cloud = Map()
    pq = PriorityQueue()

    for node in graph._nodes:
        node_id = node.get_id()
        if node_id == origin:
            weights.insert_kv(node_id, 0)
        else:
            weights.insert_kv(node_id, float('inf'))
        pq.insert(weights.find(node_id), node_id)

    while not pq.is_empty():
        c_weight, c_node_id = pq.remove_min_with_key()
        cloud.insert_kv(c_node_id, c_weight)
        valid_locations.append(Entry(c_node_id, c_weight))

        # Explore neighbors
        for node, weight in graph.get_neighbours(c_node_id):
            node_id = node.get_id()
            if cloud.find(node_id) is None:
                p_weight = weights.find(c_node_id) + weight

                # Check if the path through c_node_id is better
                if p_weight < weights.find(node_id):
                    old_weight = weights.find(node_id)
                    # Update with old and new weight
                    pq.update(old_weight, p_weight, node_id)
                    # Update the weight in the map
                    weights.insert_kv(node_id, p_weight)

    return valid_locations


def dfs_traversal(
    graph: Graph | LatticeGraph, origin: int, goal: int
) -> tuple[DynamicArray, DynamicArray]:
    """
    Task 2.3: Depth First Search **** COMP7505 ONLY ****
    COMP3506 students can do this for funsies.

    @param: graph
      The general graph or lattice graph to process
    @param: origin
      The ID of the node from which to start traversal
    @param: goal
      The ID of the target node

    @returns: tuple[DynamicArray, DynamicArray]
      1. The ordered path between the origin and the goal in node IDs
      (or an empty DynamicArray if no path exists);
      2. The IDs of all nodes in the order they were visited.

    """
    # Stores the keys of the nodes in the order they were visited
    visited_order = DynamicArray()
    # Stores the path from the origin to the goal
    path = DynamicArray()

    # ALGO GOES HERE

    # Return the path and the visited nodes list
    return (path, visited_order)
