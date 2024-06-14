"""Games or Adversarial Search (Chapter 5)"""

import random
import time
from collections import namedtuple

import numpy as np

"""
// Student Info
// ------------
//
// Name : <mahdi beigahmadi>
// St.# : <301570853>
// Email: <mba188@sfu.ca>
//
"""
GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')


def gen_state(move='(1, 1)', to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    """
        move = the move that has lead to this state,
        to_move=Whose turn is to move
        x_position=positions on board occupied by X player,
        o_position=positions on board occupied by O player,
        (optionally) number of rows, columns and how many consecutive X's or O's required to win,
    """
    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, move=move, utility=0, board=board, moves=moves)


# ______________________________________________________________________________
# MinMax Search
def minmax(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""
    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)), default=None)


def minmax_cutoff(game, state, depth):
    """Given a state in a game, calculate the best move by searching
    forward to the given cutoff depth. At that level, use the evaluation function."""
    memoryForOptimization = {}

    def convertTheStatesIntoHashes(state, depth):
        """Create a hashable representation of the state and depth."""
        return state.to_move, tuple(sorted(state.board.items())), state.utility, depth

    def max_value(state, d):
        if game.terminal_test(state):
            return game.utility(state, game.to_move(state))
        if d == 0:
            return game.eval1(state)
        hashState = convertTheStatesIntoHashes(state, d)
        if hashState in memoryForOptimization:
            return memoryForOptimization[hashState]
        v = -np.inf
        for a in sorted(game.actions(state), key=lambda x: heuristic(game, state, x), reverse=True):
            v = max(v, min_value(game.result(state, a), d - 1))
        memoryForOptimization[hashState] = v
        return v

    def min_value(state, d):
        if game.terminal_test(state):
            return game.utility(state, game.to_move(state))
        if d == 0:
            return game.eval1(state)
        hashState = convertTheStatesIntoHashes(state, d)
        if hashState in memoryForOptimization:
            return memoryForOptimization[hashState]
        v = np.inf
        for a in sorted(game.actions(state), key=lambda x: heuristic(game, state, x)):
            v = min(v, max_value(game.result(state, a), d - 1))
        memoryForOptimization[hashState] = v
        return v

    def heuristic(game, state, move):
        nextState = game.result(state, move)
        return game.utility(nextState, game.to_move(state))

    highestScore = -np.inf
    bestPossibleAction = None
    for a in game.actions(state):
        value = min_value(game.result(state, a), depth - 1)
        if value > highestScore:
            highestScore = value
            bestPossibleAction = a
    return bestPossibleAction


# ______________________________________________________________________________


def alpha_beta(game, state):
    """Search game to determine best action; use alpha-beta pruning.
     this version searches all the way to the leaves."""
    player = game.to_move(state)

    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player), None
        value, move = -np.inf, None
        for action in game.actions(state):
            secondaryValue, _ = min_value(game.result(state, action), alpha, beta)
            if secondaryValue > value:
                value, move = secondaryValue, action
                alpha = max(alpha, value)
            if value >= beta:
                return value, move
        return value, move

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player), None
        value, move = np.inf, None
        for action in game.actions(state):
            secondaryValue, _ = max_value(game.result(state, action), alpha, beta)
            if secondaryValue < value:
                value, move = secondaryValue, action
                beta = min(beta, value)
            if value <= alpha:
                return value, move
        return value, move

    _, bestMoveForAlphaBeta = max_value(state, -np.inf, np.inf)
    return bestMoveForAlphaBeta


def alpha_beta_cutoff(game, state, depth):
    """Search game to determine best action; use alpha-beta pruning.
       This version cuts off search at a given depth and uses an evaluation function."""

    def max_value(state, alpha, beta, depth):
        if game.terminal_test(state) or depth == 0:
            return game.eval1(state), None
        value, move = -np.inf, None
        for action in game.actions(state):
            newValue, _ = min_value(game.result(state, action), alpha, beta, depth - 1)
            if newValue > value:
                value, move = newValue, action
            alpha = max(alpha, value)
            if value >= beta:
                break
        return value, move

    def min_value(state, alpha, beta, depth):
        if game.terminal_test(state) or depth == 0:
            return game.eval1(state), None
        value, move = np.inf, None
        for action in game.actions(state):
            newValue, _ = max_value(game.result(state, action), alpha, beta, depth - 1)
            if newValue < value:
                value, move = newValue, action
            beta = min(beta, value)
            if value <= alpha:
                break
        return value, move

    _, bestMove = max_value(state, -np.inf, np.inf, depth)
    return bestMove


# ______________________________________________________________________________
# Players for Games
def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move


def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


def alpha_beta_player(game, state):
    """Uses alpha-beta pruning with minmax, or with cutoff version, for AI player."""
    if game.timer < 0:
        return alpha_beta(game, state)

    start = time.perf_counter()
    depth = 1
    bestPossibleMove = None

    while True:
        currentMove = alpha_beta_cutoff(game, state, depth)
        print(f"Searching at depth: {depth}, Time elapsed: {time.perf_counter() - start}s")

        if currentMove is not None:
            bestPossibleMove = currentMove

        if time.perf_counter() - start > game.timer:
            print(f"Breaking out of loop after {depth} depths due to time limit.")
            break

        depth += 1

    print(f"Iterative deepening reached depth: {depth}")
    return bestPossibleMove


def minmax_player(game, state):
    """Uses minimax or minimax with cutoff depth for AI player."""
    if game.timer < 0:
        return minmax(game, state)

    start = time.perf_counter()
    depth = 1
    bestPossibleMove = None

    while True:
        print(f"Searching at depth: {depth}")
        currentMove = minmax_cutoff(game, state, depth)
        if currentMove is not None:
            bestPossibleMove = currentMove
        if time.perf_counter() - start > game.timer:
            break
        depth += 1

    print(f"Iterative deepening reached depth: {depth}")
    return bestPossibleMove


# ______________________________________________________________________________
# base class for Games

class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))


class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to_move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
    depth = -1 means max search tree depth to be used."""

    def __init__(self, size=3, k=3, t=-1):
        self.size = size
        if k <= 0:
            self.k = size
        else:
            self.k = k
        self.d = -1  # d is cutoff depth. Default is -1 meaning no depth limit. It is controlled usually by timer
        self.maxDepth = size * size  # max depth possible is width X height of the board
        self.timer = t  # timer  in seconds for opponent's search time limit. -1 means unlimited
        moves = [(x, y) for x in range(1, size + 1)
                 for y in range(1, size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def reset(self):
        moves = [(x, y) for x in range(1, self.size + 1)
                 for y in range(1, self.size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    @staticmethod
    def switchPlayer(player):
        assert (player == 'X' or player == 'O')
        return 'O' if player == 'X' else 'X'

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        try:
            moves = list(state.moves)
            moves.remove(move)
        except (ValueError, IndexError, TypeError) as e:
            print("exception: ", e)

        return GameState(to_move=self.switchPlayer(state.to_move), move=move,
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or lost or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(0, self.size):
            for y in range(1, self.size + 1):
                print(board.get((self.size - x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If player wins with this move, return k if player is 'X' and -k if 'O' else return 0."""
        if (self.k_in_row(board, move, player, (0, 1), self.k) or
                self.k_in_row(board, move, player, (1, 0), self.k) or
                self.k_in_row(board, move, player, (1, -1), self.k) or
                self.k_in_row(board, move, player, (1, 1), self.k)):
            return self.k if player == 'X' else -self.k
        else:
            return 0

    # evaluation function, version 1

    def eval1(self, state):
        def evaluateScoreLines(segment, player):
            opponent = 'O' if player == 'X' else 'X'
            playerCount = segment.count(player)
            opponentCount = segment.count(opponent)

            scores = {index: 10 ** index for index in range(1, self.k)}

            if playerCount > 0 and opponentCount == 0:
                return scores.get(playerCount, 0)
            elif opponentCount > 0 and playerCount == 0:
                return -scores.get(opponentCount, 0)

            return 0

        utility = 0
        size = self.size
        board = state.board
        gameLines = []

        for i in range(1, size + 1):
            row = [board.get((i, j), '.') for j in range(1, size + 1)]
            column = [board.get((j, i), '.') for j in range(1, size + 1)]
            gameLines.append(row)
            gameLines.append(column)

        for start in range(1, size - self.k + 2):
            primaryDiag1 = [board.get((start + i, i + 1), '.') for i in range(size - start + 1)]
            secondaryDiag1 = [board.get((i + 1, start + i), '.') for i in range(size - start + 1)]
            primaryDiag2 = [board.get((i + 1, start + size - 1 - i), '.') for i in range(size - start + 1)]
            secondaryDiag2 = [board.get((start + i, size - i), '.') for i in range(size - start + 1)]

            if len(primaryDiag1) >= self.k:
                gameLines.append(primaryDiag1)
            if len(secondaryDiag1) >= self.k:
                gameLines.append(secondaryDiag1)
            if len(primaryDiag2) >= self.k:
                gameLines.append(primaryDiag2)
            if len(secondaryDiag2) >= self.k:
                gameLines.append(secondaryDiag2)

        currentPlayer = state.to_move
        for line in gameLines:
            utility += evaluateScoreLines(line, currentPlayer)

        return utility

    @staticmethod
    def k_in_row(board, pos, player, dir, k):
        """Return true if there is a line of k cells in direction dir including position pos on board for player."""
        (delta_x, delta_y) = dir
        x, y = pos
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = pos
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= k
