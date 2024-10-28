"""
Skeleton for COMP3506/7505 A1, S2, 2024
The University of Queensland
Joel Mackenzie and Vladimir Morozov
"""

# so we can hint Node get_next
from __future__ import annotations

from typing import Any


class Node:
    """
    A simple type to hold data and a next pointer
    """

    def __init__(self, data: Any) -> None:
        self._data = data  # This is the payload data of the node
        self._next = None  # This is the "next" pointer to the next Node
        self._prev = None  # This is the "previous" pointer to the previous Node

    def set_data(self, data: Any) -> None:
        self._data = data

    def get_data(self) -> Any:
        return self._data

    def set_next(self, node: Node) -> None:
        self._next = node

    def get_next(self) -> Node | None:
        return self._next

    def set_prev(self, node: Node) -> None:
        self._prev = node

    def get_prev(self) -> Node | None:
        return self._prev


class DoublyLinkedList:
    """
    Your doubly linked list code goes here.
    Note that any time you see `Any` in the type annotations,
    this refers to the "data" stored inside a Node.

    [V3: Note that this API was changed in the V3 spec] 
    """

    def __init__(self) -> None:
        # You probably need to track some data here...
        self._size = 0
        self._head = None
        self._tail = None
        self._reversed = False

    def __str__(self) -> str:
        """
        A helper that allows you to print a DoublyLinkedList type
        via the str() method.
        """
        if self._reversed:
            current_node = self._tail
        else:
            current_node = self._head
        data_str = ""
        if current_node != None:
            while (True):
                # print(current_node.get_data())
                if self._reversed:
                    if current_node.get_prev() == None:
                        data_str += current_node.get_data()
                        break
                    else:
                        data_str += f"{current_node.get_data()} <-> "
                        current_node = current_node.get_prev()
                else:
                    if current_node.get_next() == None:
                        data_str += f"{current_node.get_data()}"
                        break
                    else:
                        data_str += f"{current_node.get_data()} <-> "
                        current_node = current_node.get_next()
        return data_str

    """
    Simple Getters and Setters below
    """

    def get_size(self) -> int:
        """
        Return the size of the list.
        Time complexity for full marks: O(1)
        """
        return self._size

    def get_head(self) -> Any | None:
        """
        Return the data of the leftmost node in the list, if it exists.
        Time complexity for full marks: O(1)
        """
        if self._reversed:
            if self._tail == None:
                return None
            return self._tail.get_data()
        if self._head == None:
            return None
        return self._head.get_data()

    def set_head(self, data: Any) -> None:
        """
        Replace the leftmost node's data with the given data.
        If the list is empty, do nothing.
        Time complexity for full marks: O(1)
        """
        if self._reversed:
            if self._tail != None:
                self._tail.set_data(data)
                # self._size += 1
        else:
            if self._head != None:
                self._head.set_data(data)
                # self._size += 1

    def get_tail(self) -> Any | None:
        """
        Return the data of the rightmost node in the list, if it exists.
        Time complexity for full marks: O(1)
        """
        if self._reversed:
            if self._head == None:
                return None
            return self._head.get_data()
        if self._tail == None:
            return None
        return self._tail.get_data()

    def set_tail(self, data: Any) -> None:
        """
        Replace the rightmost node's data with the given data.
        If the list is empty, do nothing.
        Time complexity for full marks: O(1)
        """
        if self._reversed:
            if self._head != None:
                self._head.set_data(data)
                # self._size += 1
        else:
            if self._tail != None:
                self._tail.set_data(data)
                # self._size += 1

    """
    More interesting functionality now.
    """

    def insert_to_front(self, data: Any) -> None:
        """
        Insert the given data to the front of the list.
        Hint: You will need to create a Node type containing
        the given data.
        Time complexity for full marks: O(1)
        """
        if self._reversed:
            new_tail = Node(data)
            if self._tail == None:
                if self._head != None:
                    self._tail = new_tail
                    self._head.set_next(self._tail)
                else:
                    self._head = new_tail
                    self._tail = new_tail
            else:
                self._tail.set_next(new_tail)
                new_tail.set_prev(self._tail)
                self._tail = new_tail
            self._size += 1
        else:
            new_head = Node(data)
            # print(new_head.get_data)
            if self._head == None:
                self._head = new_head
                self._tail = new_head
            else:
                self._head.set_prev(new_head)
                # new_head.set_prev(None)
                new_head.set_next(self._head)
                self._head = new_head
            self._size += 1

    def insert_to_back(self, data: Any) -> None:
        """
        Insert the given data (in a node) to the back of the list
        Time complexity for full marks: O(1)
        """
        if self._reversed:
            new_head = Node(data)
            # print(new_head.get_data)
            if self._head == None:
                self._head = new_head
                self._tail = new_head
            else:
                self._head.set_prev(new_head)
                # new_head.set_prev(None)
                new_head.set_next(self._head)
                self._head = new_head
            self._size += 1
        else:
            new_tail = Node(data)
            if self._tail == None:
                if self._head != None:
                    self._tail = new_tail
                    self._head.set_next(self._tail)
                else:
                    self._head = new_tail
                    self._tail = new_tail
            else:
                self._tail.set_next(new_tail)
                new_tail.set_prev(self._tail)
                self._tail = new_tail
            self._size += 1

    def remove_from_front(self) -> Any | None:
        """
        Remove the front node, and return the data it holds.
        Time complexity for full marks: O(1)
        """
        if self._reversed:
            if self._tail == None:
                return None
            data = self._tail.get_data()
            if self._size > 1:
                new_tail = self._tail.get_prev()
                new_tail.set_next(None)
                self._tail.set_prev(None)
                self._tail = new_tail
            else:
                self._tail.set_next(None)
                self._tail.set_prev(None)
                self._head = None
                self._tail = None
            self._size -= 1
            return data
        else:
            if self._head == None:
                return None
            data = self._head.get_data()
            if self._size > 1:
                new_head = self._head.get_next()
                new_head.set_prev(None)
                self._head.set_next(None)
                self._head = new_head
            else:
                self._head.set_next(None)
                self._head.set_prev(None)
                self._head = None
                self._tail = None
            self._size -= 1
            return data

    def remove_from_back(self) -> Any | None:
        """
        Remove the back node, and return the data it holds.
        Time complexity for full marks: O(1)
        """
        if self._reversed:
            data = self._head.get_data()
            if self._size > 1:
                new_head = self._head.get_next()
                new_head.set_prev(None)
                self._head.set_next(None)
                self._head = new_head
            else:
                self._head.set_next(None)
                self._head.set_prev(None)
                self._head = None
                self._tail = None
            self._size -= 1
            return data
        else:
            if self._tail == None:
                return None
            data = self._tail.get_data()
            if self._size > 1:
                new_tail = self._tail.get_prev()
                new_tail.set_next(None)
                self._tail.set_prev(None)
                self._tail = new_tail
            else:
                self._tail.set_next(None)
                self._tail.set_prev(None)
                self._head = None
                self._tail = None
            self._size -= 1
            return data

    def find_element(self, elem: Any) -> bool:
        """
        Looks at the data inside each node of the list and returns True
        if a match is found; False otherwise.
        Time complexity for full marks: O(N)
        """
        found = False
        if not self._reversed:
            current_node = self._head
        else:
            current_node = self._tail
        while current_node is not None:
            if current_node.get_data() == elem:
                return True
            if not self._reversed:
                current_node = current_node.get_next()
            else:
                current_node = current_node.get_prev()  
        return found
        
    def find_and_remove_element(self, elem: Any) -> bool:
        """
        Looks at the data inside each node of the list; if a match is
        found, this node is removed from the linked list, and True is returned.
        False is returned if no match is found.
        Time complexity for full marks: O(N)
        """
        found = False
        # if not self._reversed:
        current_node = self._head
        # else:
            # current_node = self._tail
        while current_node is not None:
            if current_node.get_data() == elem:
                found = True
                if current_node == self._head:
                    self.remove_from_front()
                elif current_node == self._tail:
                    self.remove_from_back()
                else:
                    current_node.get_prev().set_next(current_node.get_next())
                    current_node.get_next().set_prev(current_node.get_prev())
                    current_node.set_next(None)
                    current_node.set_prev(None)
                    self._size -= 1
                break
            else:
                # if not self._reversed:
                    current_node = current_node.get_next()
                # else:
                    # current_node = current_node.get_prev()
        return found

    def reverse(self) -> None:
        """
        Reverses the linked list
        Time complexity for full marks: O(1)
        """
        self._reversed = not self._reversed

    def return_head(self) -> Node:
        return self._head

    