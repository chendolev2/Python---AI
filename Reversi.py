import numpy as np
import random
import sys
import copy
import heapq


BOARD_SIZE = 8   # How many rows and columns in the board, shuold be even
NUMBER_OF_DISKS = 64
HOW_MANY_DISK_BEGINING = 4 # With how many disks to start the game. should be a square number and even
# How the changes on board wil be drawed
PLAYER_ONE_SIGN = "x"
PLAYER_TWO_SIGN = "o"
EMPTY_CELL_SIGN = "-"

# Draw board by boolean values and players sign
def print_board(board):
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j]:  # Player one
                print(PLAYER_ONE_SIGN, end="")
            elif board[i][j] == False:
                print(PLAYER_TWO_SIGN, end="")
            else:  # Empty space in board without disks
                print(EMPTY_CELL_SIGN, end="")
        print()  # Start a new line in the nex row


# Initilize the board with the disks of player one and two represented by boolean values
# player one - true, player two - false
def reset_board(board):
    middle_index = BOARD_SIZE // 2 - 1
    board[middle_index][middle_index] = True
    board[middle_index + 1][middle_index + 1] = True
    board[middle_index][middle_index + 1] = False
    board[middle_index + 1][middle_index] = False


# Print result, state and board status after every player's turn
def display(board_before_change, board_after_change, state_num, player1_turn, coordinates, disk_counts):
    player = 1 if player1_turn else 2  # Number of player
    total_disks = disk_counts[1] + disk_counts[2]

    # Print board before change
    print("State", state_num)
    print_board(board_before_change)

    # Print board after change
    print("State", state_num + 1, ", Player", player, "moved, Action: A disk has been added to the coordinate" ,coordinates)
    print_board(board_after_change)
    print("Result - Player 1:", disk_counts[1], "disks, Player 2:", disk_counts[2], "disks, Total:", total_disks, "disks")


# Calculate which opponent disks should be flipped after the other player adds one disk
def coordinate_impacts(board_after_change, new_coordinate, player1_turn, directions):
    x_coordinate, y_coordinate = new_coordinate
    opponent_turn = not player1_turn
    flip_coordinates = []


    for direction_x, direction_y in directions:
        x, y = x_coordinate, y_coordinate
        temp_flips = []  # Temporarily store disks to flip
        x += direction_x
        y += direction_y

        # Go in the given direction
        while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            if board_after_change[x][y] == opponent_turn:  # Opponent disk
                temp_flips.append((x, y))
            elif board_after_change[x][y] == player1_turn:  # Player disk bounds the line
                flip_coordinates.extend(temp_flips)  # Add all valid flips
                break
            else:  # Empty cell or invalid
                break
            x += direction_x
            y += direction_y

    return flip_coordinates  # Return all flippable coordinates


def display_all_possible_actions(board_before_change, player1_turn, flip_coordinates, disk_counts, current_state, directions):
    action_num = 0  # Present the user the number of action being analyzed
    possible_coordinates = all_possible_moves(board_before_change, player1_turn, directions)

    if not possible_coordinates:  # No valid moves available
        print("No possible moves from this state")
        return -1, -1

    for coordinate in possible_coordinates:
        i, j = coordinate[0]

        action_num += 1
        board_testing = np.copy(board_before_change)
        disk_counts_copy = copy.deepcopy(disk_counts)

        temp_flips = coordinate[1]
        flip_coordinates.clear()
        flip_coordinates.extend(temp_flips)

        # Flip the opponent_disks and the current coodrinate and count how many disks each player has
        board_testing[i][j] = player1_turn
        flip_opponent_disks(board_testing, player1_turn, flip_coordinates)

        # Update each player's disks count
        update_disk_counts(len(flip_coordinates), player1_turn,  disk_counts_copy)

        # Display the result
        print("Presents possible actions for player", 1 if player1_turn else 2)
        print("For action", action_num, "\n")
        display(board_before_change, board_testing, current_state, player1_turn, (i,j), disk_counts_copy)
        print()




# Change the boolean value of
def flip_opponent_disks(board_after_change, player1_turn, flip_coordinate):
    for i, j in flip_coordinate:
        board_after_change[i][j] = player1_turn  # Flip the disk at (i, j)


# Goes by matrix order to find valid coordinate to add disk at
def choose_valid_coordinate_by_order(flip_coordinates, board_before_change, player1_turn, directions):
    # Find all valid moves
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board_before_change[i][j] == None:  # Check if the cell is empty
                temp_flips = coordinate_impacts(board_before_change, (i, j), player1_turn, directions)
                if temp_flips:  # Can flip at least one opponent's disk
                    flip_coordinates.clear()
                    flip_coordinates.extend(temp_flips)
                    return i, j
    return -1, -1

# Find all possible moves for spesific state
def all_possible_moves(board_before_change, player1_turn, directions):
    possible_coordinates = []
    # Find all valid moves
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board_before_change[i][j] == None:  # Check if the cell is empty
                temp_flips = coordinate_impacts(board_before_change, (i, j), player1_turn, directions)
                if temp_flips:  # Can flip at least one opponent's disk
                    possible_coordinates.append(((i, j), temp_flips))  # Add valid coordinate to the list

    return possible_coordinates

# Choose coordinate randomly from list of all possible coordinates
def choose_coordinate_randomly(flip_coordinates, board_before_change, player1_turn, directions):
    possible_coordinates = all_possible_moves(board_before_change, player1_turn, directions)

    if not possible_coordinates:   # No valid moves available
        return -1, -1

    (i_random, j_random), chosen_flips = random.choice(possible_coordinates)   # Choose random ccoordinate
    # Update flip_coordinates with the chosen move's impacts
    flip_coordinates.clear()
    flip_coordinates.extend(chosen_flips)

    return i_random, j_random

# Updates each player's disks number
def update_disk_counts(flipped_disks_num, player1_turn, disk_counts):
    if player1_turn:
        # + 1 is for the current disk that has been added and caused other disks to flip
        disk_counts[1] += (flipped_disks_num + 1)
        disk_counts[2] -= flipped_disks_num
    else:  # Player 2 turn
        disk_counts[1] -= flipped_disks_num
        disk_counts[2] += (flipped_disks_num + 1)


# Check how many valid moves the current player has. If he has more moves, he is closer to wininig
def heuristic1_how_many_moves(board, player1_turn, directions):
    possible_moves = all_possible_moves(board, player1_turn, directions)
    return len(possible_moves)

# How many current player's disks are next to an empty space (cell)
def heuristic2_frontier_disks(board, player1_turn, directions):
    frontier_count = 0
    # Scans all board
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == player1_turn:
                # Check spaces for frontier disks
                for direction_x,direction_y  in directions:
                    frontier_x, frontier_y = i + direction_x, j + direction_y
                    if 0 <= frontier_x < BOARD_SIZE and 0 <= frontier_y < BOARD_SIZE and board[frontier_x][frontier_y] is None:
                        frontier_count += 1
                        break
    return frontier_count

# Use huristic function in order to find the best move for the current player
def choose_best_move_with_heuristic_func(board, player1_turn, directions, heuristic_fn):
    possible_moves = all_possible_moves(board, player1_turn, directions)
    if not possible_moves:  # No valid moves available
        return -1, -1, []

    # Build max heap
    heap = []
    for (i, j), flips in possible_moves:
        # Calculate the heuristic score for this move
        heur_value = heuristic_fn(board, player1_turn, directions)

        # Push the move onto the heap with a negated score for max-heap logic
        # Negative value since the heap module works only with min heaps
        heapq.heappush(heap, (-heur_value, (i, j), flips))

    # Pop the move with the highest score (simulated max-heap)
    negated_score, best_move, best_flips = heapq.heappop(heap)
    return best_move[0], best_move[1], best_flips


# Choose the huristic function to operate, based on the parameters
def operate_huristic_func(board_before_change, player1_turn, directions, parameters):
    func1_to_operate = heuristic1_how_many_moves if parameters[0] == "H1" else heuristic2_frontier_disks
    if len(parameters) == 1:
        coor_x, coor_y, flips = choose_best_move_with_heuristic_func(board_before_change, player1_turn, directions, func1_to_operate)

    if len(parameters) == 2:
        func2_to_operate = heuristic1_how_many_moves if parameters[1] == "H1" else heuristic2_frontier_disks
        if player1_turn:  # Operate the function in the first parameter
            coor_x, coor_y, flips = choose_best_move_with_heuristic_func(board_before_change, player1_turn, directions, func1_to_operate)
        else:   # Other player's turn - operate the function in the second parameter
            coor_x, coor_y, flips = choose_best_move_with_heuristic_func(board_before_change, player1_turn, directions, func2_to_operate)

    return coor_x, coor_y, flips

# Calculate the best move for the current player, looking a few steps forward
def minimax(board, depth, player1_turn, directions, heuristic_fn, maximizing_player, disk_counts):
    if depth == 0 or is_game_over(board, disk_counts):
        return heuristic_fn(board, player1_turn, directions), None

    possible_moves = all_possible_moves(board, player1_turn, directions)

    if not possible_moves:  # If there are no possible moves
        return heuristic_fn(board, player1_turn, directions), None

    # If maximizing_player, initialize to low value at first
    # Minimizing player aims to minimize the the value, depends which player's turn
    best_value = -float('inf') if maximizing_player else float('inf')
    best_move = None

    # Simulate each move
    for (i, j), flips in possible_moves:
        # Create a copy of the board
        temp_board = np.copy(board)
        flip_opponent_disks(temp_board, player1_turn, flips)
        temp_board[i][j] = player1_turn

        # Calculate Minimax for the next move
        value, move = minimax(temp_board, depth - 1, not player1_turn, directions, heuristic_fn, not maximizing_player, disk_counts)

        if maximizing_player:
            if value > best_value:
                best_value = value
                best_move = (i, j)
        else:
            if value < best_value:
                best_value = value
                best_move = (i, j)

    return best_value, best_move



# Check if one of the players runs out of disks or board is full
def is_game_over(board_after_change, disk_counts):
    # Check if one of the players runs out of disks and can't flip anymore
    if disk_counts[1] == 0 or disk_counts[2] == 0:
        return True
    # Check if board is full
    for row in board_after_change:
        if None in row:
            return False
    return True

# Prints who the winner or if it's a tie
def announce_winner(disk_counts):
    if disk_counts[1] > disk_counts[2]:
        print("Player 1 is the winner")
    elif disk_counts[1] < disk_counts[2]:
        print("Player 2 is the winner")
    else:  # Both have the same amounr of disks
        print("It's a tie")


def main(parameters):
    board_before_change = np.full((BOARD_SIZE, BOARD_SIZE), None, dtype=object)  # Initializing the board
    reset_board(board_before_change)  # Places the starting disks on board
    board_after_change = np.copy(board_before_change)

    state_num = 0  # How many times did the board change during the game?
    player1_turn = True  # If false it's player2_turn
    # How many disks for  each player = half amount of disks at the begining
    disk_counts = {   1: int(HOW_MANY_DISK_BEGINING / 2), 2: int(HOW_MANY_DISK_BEGINING / 2)}

    # Check all 8 directions: row, column, and diagonals
    directions = [
        (1, 0),   # Right
        (-1, 0),  # Left
        (0, 1),   # Down
        (0, -1),  # Up
        (1, 1),   # Diagonal down-right
        (-1, -1), # Diagonal up-left
        (1, -1),  # Diagonal down-left
        (-1, 1)   # Diagonal up-right
     ]

    player1_can_play = True
    player2_can_play = True
    ahead = 0


    while not is_game_over(board_after_change, disk_counts):
        # Find a valid coordinate to add a disk to
        flip_coordinates = []
        total_disks_before_change = disk_counts[1] + disk_counts[2]
        # Choose move based on the mode

        # Need to operate huristic function or 2 huristic functions
        if parameters[0] == "H1" or parameters[0] == "H2":
            if len(parameters) != 3:
                if state_num == 0:
                    # First step is choosing the first coordinate in order
                    valid_coor_x, valid_coor_y = choose_valid_coordinate_by_order(flip_coordinates, board_before_change,
                                                                                  player1_turn, directions)
                else:
                    valid_coor_x, valid_coor_y, flip_coordinates = operate_huristic_func(board_before_change, player1_turn, directions, parameters)

            else:   # Operate min max
                if state_num == 0:
                    # First step is choosing the first coordinate in order
                    valid_coor_x, valid_coor_y = choose_valid_coordinate_by_order(flip_coordinates, board_before_change,
                                                                                  player1_turn, directions)
                else:
                    func_name = heuristic1_how_many_moves if parameters[1] == "H1" else heuristic2_frontier_disks
                    ahead_value = int(parameters[2])  # Get ahead depth from parameters
                    maximizing_player = player1_turn  # Start with the player who's turn it is
                    # Call minimax
                    best_value, best_move = minimax(board_before_change, ahead_value, player1_turn, directions,
                                                    func_name, maximizing_player, disk_counts)
                    if best_move != None:
                        valid_coor_x, valid_coor_y = best_move


        else:
            mode, n_states = parameters
            if mode == "--displayAllActions":
                if n_states <= total_disks_before_change:
                    display_all_possible_actions(board_before_change, player1_turn, flip_coordinates, disk_counts, state_num, directions)
                    break
                else:
                    valid_coor_x, valid_coor_y = choose_valid_coordinate_by_order(flip_coordinates, board_before_change,
                                                                                  player1_turn, directions)

            elif mode == "--methodical":
                if state_num < n_states:
                    valid_coor_x, valid_coor_y = choose_valid_coordinate_by_order(flip_coordinates, board_before_change, player1_turn, directions)
                else:
                    break


            else:   # mode == "--random"
                if state_num == 0:
                    # First step is choosing the first coordinate in order
                    valid_coor_x, valid_coor_y = choose_valid_coordinate_by_order(flip_coordinates, board_before_change,
                                                                                  player1_turn, directions)
                elif state_num < n_states:
                    valid_coor_x, valid_coor_y = choose_coordinate_randomly(flip_coordinates, board_before_change, player1_turn, directions)

                else:
                    break


        # If there is no valid move to a player:
        if (valid_coor_x, valid_coor_y) == (-1, -1):
            player =  1 if player1_turn is True else 2  # Number of player
            print("No legal move for player", player,". Turn pass ")
            if player1_turn:
                player1_can_play = False
            else:
                player2_can_play = False
            player1_turn = not player1_turn  # change turn to the other player
            # check if the game is stuck
            if not player1_can_play and not player2_can_play:
                break
            else:
                continue


        # Everything is OK continue
        # Flip the opponent_disks and the current coodrinate and count how many disks each player has
        board_after_change[valid_coor_x][valid_coor_y] = player1_turn
        flip_opponent_disks(board_after_change, player1_turn, flip_coordinates)

        # Update each player's disks count
        update_disk_counts(len(flip_coordinates), player1_turn,  disk_counts)

        # Display the result
        valid_coordinate = valid_coor_x, valid_coor_y
        if parameters[0] != "--displayAllActions":
            display(board_before_change, board_after_change, state_num, player1_turn, valid_coordinate, disk_counts)
        board_before_change = np.copy(board_after_change)
        state_num += 1
        player1_turn = not player1_turn # change turn to the other player

    # End of loop
    if parameters[0] != "--displayAllActions":
        if len(parameters) == 2 and parameters[1] not in ["H1", "H2"] and state_num >= int(parameters[1]):
            print("Game stopped after displaying", n_states, "states.")
        else:
            announce_winner(disk_counts)



if __name__ == "__main__":
    # Read parameters from command-line arguments

   if len(sys.argv) == 2:
        if sys.argv[1] in ["H1", "H2"]:
            main((sys.argv[1],))
        else:
            print("Invalid function name: ", sys.argv[1])
            sys.exit(1)

   elif len(sys.argv) == 3:
        if sys.argv[1] in ["H1", "H2"] and sys.argv[2] in ["H1", "H2"]:
            main((sys.argv[1], sys.argv[2]))

        else:    # Probably a mode
            mode = sys.argv[1]
            if mode not in ["--displayAllActions", "--methodical", "--random"]:
                print("Invalid mode or function: ", mode)
                sys.exit(1)

            try:
                n_states = int(sys.argv[2])  # Number of states or games to simulate
            except ValueError:
                print("The second argument must be an integer.")
                sys.exit(1)

            # Call main with the parsed parameters
            main((mode, n_states))

   elif len(sys.argv) == 4:
       if sys.argv[1] not in ["H1", "H2"] or sys.argv[2] != "--ahead":
           print("Wrong huristic function or method")

       # Check if the third argument is an integer greater than 0
       try:
           ahead_value = int(sys.argv[3])
           if ahead_value <= 0:
               print("The value after '--ahead' must be an integer greater than 0.")
               sys.exit(1)
       except ValueError:
              print("The value after '--ahead' must be a valid integer.")
              sys.exit(1)

       main((sys.argv[1], sys.argv[2], sys.argv[3]))

   else:   # Too much parameters
       print("Too much parameters")



