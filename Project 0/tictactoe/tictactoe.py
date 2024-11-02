"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Return X if it is an empty board
    if board == initial_state():
        return X

    count_x, count_o = 0, 0
    for row in board:
        for cell in row:
            if cell == X:
                count_x += 1
            elif cell == "O":
                count_o += 1
    # Return player who has the next turn
    if count_x == count_o:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    # Each empty cell is a possible move
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Check for validity of the action
    if action not in actions(board):
        raise ValueError
    # Create a deepcopy of the board
    board_copy = copy.deepcopy(board)
    # Play the next move
    board_copy[action[0]][action[1]] = player(board)
    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        # Check for winner horizontally
        if board[i][0] == board[i][1] == board[i][2]:
            if board[i][0] != EMPTY:
                return board[i][0]
        # Check for winner vertically
        if board[0][i] == board[1][i] == board[2][i]:
            if board[0][i] != EMPTY:
                return board[0][i]
    # Check for winner diagonally
    if (
        board[0][0] == board[1][1] == board[2][2]
        or board[0][2] == board[1][1] == board[2][0]
    ):
        if board[1][1] != EMPTY:
            return board[1][1]
    # In case of an incomplete game or a tie, return None
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # If there is a winner, return True
    if winner(board) in [X, O]:
        return True
    count_empty = 0
    # Count number of empty cells
    for row in board:
        for cell in row:
            if cell == EMPTY:
                count_empty += 1
    # If the board is filled, return True (Checking for a tie)
    if count_empty == 0:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    player = winner(board)
    # Return the value case-wise
    if player == X:
        return 1
    elif player == O:
        return -1
    else:
        return 0


def max_value(board):
    """
    Returns the maximum value of utility for an action from the current board assuming optimal play.
    """
    if terminal(board):
        return utility(board)
    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v


def min_value(board):
    """
    Returns the minimum value of utility for an action from the current board assuming optimal play.
    """
    if terminal(board):
        return utility(board)
    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # If game is over, return None
    if terminal(board):
        return None
    optimal_actions = []
    current_player = player(board)
    # If it is X's turn, return the action with maximum value
    if current_player == X:
        value = max_value(board)
        for action in actions(board):
            if min_value(result(board, action)) == value:
                optimal_actions.append(action)
    # If it is O's turn, return the action with minimum value
    elif current_player == O:
        value = min_value(board)
        for action in actions(board):
            if max_value(result(board, action)) == value:
                optimal_actions.append(action)
    # Prioritize actions which end the game
    for action in optimal_actions:
        if terminal(result(board, action)):
            return action
    return optimal_actions[0]
