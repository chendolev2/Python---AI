import numpy as np
import queue
import heapq as heap
import sys


GOAL_STATE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
PUZZLE_SIZE = 9
SEPARATED_LINE = "----------------------------------------------------------"


# we assume every puzzle we get has only 9 values
# checks if the transition from one puzzle to another (depending on the index) is legal.
def is_valid_move(i, index_of_zero):  # i is the index we want to move its value to the gap area
    return (index_of_zero == 0 and i in [1, 3]) or (index_of_zero == 1 and i in [0, 2, 4]) or \
           (index_of_zero == 2 and i in [1, 5]) or (index_of_zero == 3 and i in [0, 4]) or \
           (index_of_zero == 4 and i in [1, 3, 5, 7]) or (index_of_zero == 5 and i in [2, 4, 8]) or \
           (index_of_zero == 6 and i in [3, 7]) or (index_of_zero == 7 and i in [4, 6, 8]) or \
           (index_of_zero == 8 and i in [5, 7])


# prints all the steps (indexes) from the start to the end
#     we need to switch with the gap index in order to achieve the goal puzzle
def print_solution_path(puzzle, node_parent_dict):
    path = []
    current_puzzle = tuple(puzzle)
    # while it isn't the puzzle we started from
    while current_puzzle in node_parent_dict and node_parent_dict[current_puzzle][1] != -1:
        parent_puzzle, parent_index = node_parent_dict[current_puzzle]
        path.append(parent_index)
        current_puzzle = tuple(parent_puzzle)
    print("Path:", ' '.join(map(str, reversed(path))))   # reverse the step so it will be presented from start to end
    print(SEPARATED_LINE)


# check if the current puzzle is the goal puzzle, if it is, print the step from the starting puzzle.
# document the current puzzle if it hasn't document yet in checked puzzles
def is_correct_answer(current_puzzle, expended_nodes, node_parent, checked_puzzles):
    if np.array_equal(GOAL_STATE, current_puzzle):  # the current puzzle is the goal puzzle we search
        print("Expended nodes during solution:", expended_nodes)
        print_solution_path(current_puzzle, node_parent)
        return True

    if tuple(current_puzzle) not in checked_puzzles:
        checked_puzzles.add(tuple(current_puzzle))  # mark puzzle as checked
        # pass from left to right in the array, each number in the puzzle
    return False


# Searching the goal puzzle in the puzzle's tree by scanning level by level
# each level down (as we gp deeper in the tree), there is one change in the puzzle
# (switch 0 with number in index next to it)
def bfs_search(puzzle: str):
    print("BFS")
    expended_nodes = 0  # how many puzzles have been checked until the solution was found (node = puzzle)
    puzzle_lst = np.array(list(map(int, puzzle.split())))  # make the str input to an array
    search_queue = queue.Queue()  # creates new queue
    search_queue.put(puzzle_lst)  # current puzzle we start from
    checked_puzzles = set()  # marks all the checked options
    checked_puzzles.add(tuple(puzzle_lst))  # add the start puzzle we try to solve
    node_parent = {tuple(puzzle_lst): (None, -1)}  # track the change process in the puzzle until finding the solution

    while search_queue:
        current_puzzle = search_queue.get()
        expended_nodes += 1
        # check if current puzzle is the answer
        if is_correct_answer(current_puzzle, expended_nodes, node_parent, checked_puzzles):
            return

        # pass from left to right in the array, each number in the puzzle
        index_of_zero = np.where(current_puzzle == 0)[0][0]
        for i in range(PUZZLE_SIZE):
            puzzle_change = current_puzzle.copy()
            puzzle_change[i], puzzle_change[index_of_zero] = puzzle_change[index_of_zero], puzzle_change[i]
            if is_valid_move(i, index_of_zero) and tuple(puzzle_change) not in checked_puzzles:
                search_queue.put(np.array(puzzle_change))
                node_parent[tuple(puzzle_change)] = (current_puzzle, i)

    print("No solution was found for this puzzle")
    return


# search in the puzzles tree until depth limit. In each iteration the depth limit is raised by one.
def iddfs_search(puzzle: str) -> None:
    print("IDDFS")
    expended_nodes = 0  # how many puzzles have been checked until the solution was found (node = puzzle)
    puzzle_lst = np.array(list(map(int, puzzle.split())))  # make the str input to an array
    search_queue = queue.Queue()  # creates new queue
    search_queue.put(puzzle_lst)  # current puzzle we start from
    checked_puzzles = set()  # marks all the checked options
    checked_puzzles.add(tuple(puzzle_lst))  # add the start puzzle we try to solve
    node_parent = {tuple(puzzle_lst): (None, -1)}  # track the change process in the puzzle until finding the solution
    depth_limit = 0  # each iteration a bigger depth is being checked (raised by one)
    current_depth = 0

    while search_queue:
        current_puzzle = search_queue.get()
        expended_nodes += 1
        # check if current puzzle is the answer
        if is_correct_answer(current_puzzle, expended_nodes, node_parent, checked_puzzles):
            return

        if current_depth == depth_limit:  # can't go deeper in the tree
            # check if the goal puzzle was found and in the queue
            while not search_queue.empty():
                puzzle = search_queue.get()
                expended_nodes += 1
                # check if current puzzle is the answer
                if np.array_equal(GOAL_STATE, puzzle):  # the current puzzle is the goal puzzle we search
                    print("Expended nodes during solution:", expended_nodes)
                    print_solution_path(puzzle, node_parent)
                    return

            current_depth = 0
            depth_limit += 1
            search_queue.put(puzzle_lst)  # start again from the puzzle we got as parameter
            checked_puzzles = set()  # marks all the checked options
            checked_puzzles.add(tuple(puzzle_lst))  # add the start puzzle we try to solve
            continue  # move to the next iteration

        index_of_zero = np.where(current_puzzle == 0)[0][0]
        for i in range(PUZZLE_SIZE):
            puzzle_change = current_puzzle.copy()
            puzzle_change[i], puzzle_change[index_of_zero] = puzzle_change[index_of_zero], puzzle_change[i]
            if is_valid_move(i, index_of_zero) and tuple(puzzle_change) not in checked_puzzles:
                search_queue.put(np.array(puzzle_change))
                node_parent[tuple(puzzle_change)] = (current_puzzle, i)
        current_depth += 1  # current depth is the max depth of nodes the had been checked so far

    print("No solution was found for this puzzle")
    return


#  ___________________________________________________________________________________
# Algorithms that use heuristic function  - GBFS and A*, including the definition of the heuristic function itself

# part of the heuristic function, calculates how many switched pairs there are
# pairs that if we switch them will get their right position or much closer to the wanted order in the puzzle
def calculate_inverted_pairs_value(puzzle):
    inverted_pairs = 0
    row_col_size = PUZZLE_SIZE ** 0.5

    # iterate through each pair of tiles (i, j)
    for i in range(PUZZLE_SIZE):
        for j in range(i + 1, PUZZLE_SIZE):
            # calculate current positions (row, col) for tiles i and j
            current_row_i, current_col_i = divmod(i, row_col_size)
            current_row_j, current_col_j = divmod(j, row_col_size)

            # calculate goal positions (row, col) for tiles i and j
            goal_row_i, goal_col_i = divmod(puzzle[i], row_col_size)
            goal_row_j, goal_col_j = divmod(puzzle[j], row_col_size)

            # check if tiles i and j are in the same row and need to swap
            if current_row_i == current_row_j and goal_col_i == current_col_j and goal_col_j == current_col_i:
                inverted_pairs += 1

            # check if tiles i and j are in the same column and need to swap
            if current_col_i == current_col_j and goal_row_i == current_row_j and goal_row_j == current_row_i:
                inverted_pairs += 1
    return inverted_pairs * 2


# heuristic function for both gbfs and A* search -
#  based on Manhattan distance with the sum of inverted pairs in the same row or column
#  that are in each other position or in a switched order
# this function doesn't calculate distance for A* search, it will be calculated in the A* search function
def heuristic_for_gbfs_and_a_star(puzzle):
    manhattan_distance = 0
    for i in range(PUZZLE_SIZE):
        if puzzle[i] != 0:
            current_row, current_col = divmod(i, PUZZLE_SIZE ** 0.5)
            goal_row, goal_col = divmod(puzzle[i], PUZZLE_SIZE ** 0.5)
            manhattan_distance += (abs(current_row - goal_row) + abs(current_col - goal_col))

    inverted_pairs = calculate_inverted_pairs_value(puzzle)
    return manhattan_distance + inverted_pairs


# for each checked puzzle, the distance value is calculated (Manhattan distance + max misplaced numbers in a row/col
# the function continues with the smallest value its find by using min hip  and continues with the matching puzzle until
# finding the goal puzzle)
def gbfs_search(puzzle: str) -> None:
    print("GBFS")
    expended_nodes = 0  # how many puzzles have been checked until the solution was found (node = puzzle)
    puzzle_lst = list(map(int, puzzle.split()))  # make the str input to an array
    priority_queue = []  # create priority queue to represent min heap
    heap.heappush(priority_queue,
                  (heuristic_for_gbfs_and_a_star(puzzle_lst), puzzle_lst))  # insert the starting puzzle to the queue
    checked_puzzles = set()  # marks all the checked options
    checked_puzzles.add(tuple(puzzle_lst))  # add the start puzzle we try to solve
    node_parent = {tuple(puzzle_lst): (None, -1)}  # track the change process in the puzzle until finding the solution

    while priority_queue:
        current_puzzle = heap.heappop(priority_queue)[1]
        expended_nodes += 1
        # check if current puzzle is the answer
        if is_correct_answer(current_puzzle, expended_nodes, node_parent, checked_puzzles):
            return

        index_of_zero = current_puzzle.index(0)
        for i in range(PUZZLE_SIZE):
            puzzle_change = current_puzzle.copy()
            puzzle_change[i], puzzle_change[index_of_zero] = puzzle_change[index_of_zero], puzzle_change[i]
            if is_valid_move(i, index_of_zero) and tuple(puzzle_change) not in checked_puzzles:
                heap.heappush(priority_queue, (heuristic_for_gbfs_and_a_star(puzzle_change), puzzle_change))
                node_parent[tuple(puzzle_change)] = (current_puzzle, i)

    print("No solution was found for this puzzle")
    return


# for each checked puzzle, the distance value is calculated (Manhattan distance + inverted pairs that need to be
# switched, to that value we add the value of path so far (from the starting puzzle) the function chooses the
#   puzzle with the smallest value to continue with until finding the goal puzzle
def a_star_search(puzzle: str) -> None:
    print("A*")
    expended_nodes = 0  # how many puzzles have been checked until the solution was found (node = puzzle)
    puzzle_lst = list(map(int, puzzle.split()))  # make the str input to an array
    priority_queue = []  # create priority queue to represent min heap
    heap.heappush(priority_queue,
                  (heuristic_for_gbfs_and_a_star(puzzle_lst), puzzle_lst))  # insert the starting puzzle to the queue
    checked_puzzles = set()  # marks all the checked options
    checked_puzzles.add(tuple(puzzle_lst))  # add the start puzzle we try to solve
    # track the change process in the puzzle until finding the solution
    # -1 represent there is no parent to the node, 0 represent distance
    node_parent = {tuple(puzzle_lst): (None, -1, 0)}
    node_distance = {tuple(puzzle_lst): 0}

    while priority_queue:
        current_puzzle = heap.heappop(priority_queue)[1]
        expended_nodes += 1
        # check if current puzzle is the answer
        if np.array_equal(GOAL_STATE, current_puzzle):  # the current puzzle is the goal puzzle we search
            print("Expended nodes during solution:", expended_nodes)
            print_solution_path(current_puzzle, node_parent)
            return

        if tuple(current_puzzle) not in checked_puzzles:
            checked_puzzles.add(tuple(current_puzzle))  # mark puzzle as checked

        index_of_zero = current_puzzle.index(0)
        for i in range(PUZZLE_SIZE):
            puzzle_change = current_puzzle.copy()
            puzzle_change[i], puzzle_change[index_of_zero] = puzzle_change[index_of_zero], puzzle_change[i]
            if is_valid_move(i, index_of_zero) and tuple(puzzle_change) not in checked_puzzles:
                node_parent[tuple(puzzle_change)] = (current_puzzle, i)
                node_distance[tuple(puzzle_change)] = node_distance[tuple(current_puzzle)] + 1
                heap.heappush(priority_queue, (heuristic_for_gbfs_and_a_star(puzzle_change) +
                                               node_distance[tuple(puzzle_change)], puzzle_change))

    print("No solution was found for this puzzle")
    return


# gets the parameters and runs all the functions
def main():
    input_string = ' '.join(sys.argv[1:])  # get the input string from the command line

    # call all functions with the input string
    bfs_search(input_string)
    iddfs_search(input_string)
    gbfs_search(input_string)
    a_star_search(input_string)


if __name__ == "__main__":
    main()

