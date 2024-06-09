import copy
import math
import random
import sys
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


def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


# MonteCarlo Tree Search support


class MCTS:  # Monte Carlo Tree Search implementation
    class Node:
        def __init__(self, state, par=None):
            self.state = copy.deepcopy(state)

            self.parent = par
            self.children = []
            self.visitCount = 0
            self.winScore = 0

        def getChildWithMaxScore(self):
            maxScoreChild = max(self.children, key=lambda x: x.visitCount)
            return maxScoreChild

    def __init__(self, game, state):
        self.root = self.Node(state)
        self.state = state
        self.game = game
        self.exploreFactor = math.sqrt(2)

    def isTerminalState(self, utility, moves):
        return utility != 0 or len(moves) == 0

    def monteCarloPlayer(self, timelimit=4):
        """Entry point for Monte Carlo search"""
        start = time.perf_counter()
        end = start + timelimit

        """Use timer above to apply iterative deepening"""
        while time.perf_counter() < end:
            # count = 100  # use this and the next line for debugging. Just disable previous while and enable these 2 lines
            # while count >= 0:
            # count -= 1

            # SELECT stage use selectNode()
            nodeForExploration = self.selectNode(self.root)

            # EXPAND stage
            self.expandNode(nodeForExploration)

            # SIMULATE stage using simuplateRandomPlay()
            simulationResult = self.simulateRandomPlay(nodeForExploration)

            # BACKUP stage using backPropagation
            self.backPropagation(nodeForExploration, simulationResult)

        actualTimeSpent = time.perf_counter() - start
        print(f"Time allotted: {timelimit}s, Time spent: {actualTimeSpent:.2f}s")
        winnerNode = self.root.getChildWithMaxScore()
        assert (winnerNode is not None)
        return winnerNode.state.move

    """selection stage function. walks down the tree using findBestNodeWithUCT()"""

    def selectNode(self, nd):
        node = nd
        while not self.isTerminalState(node.state.utility, node.state.moves):
            if len(node.children) == 0:
                self.expandNode(node)
            else:
                node = self.findBestNodeWithUCT(node)
        return node

    def findBestNodeWithUCT(self, nd):
        """finds the child node with the highest UCT. Parse nd's children and use uctValue() to collect uct's for the
        children....."""
        bestValue = -np.inf
        bestNode = None
        for child in nd.children:
            uctValue = self.uctValue(nd.visitCount, child.winScore, child.visitCount)
            if uctValue > bestValue:
                bestValue = uctValue
                bestNode = child
        return bestNode

    def uctValue(self, parentVisit, nodeScore, nodeVisit):
        """compute Upper Confidence Value for a node"""
        if nodeVisit == 0:
            return 0 if self.exploreFactor == 0 else sys.maxsize
        return (nodeScore / nodeVisit) + self.exploreFactor * math.sqrt(math.log(parentVisit) / nodeVisit)

    def expandNode(self, nd):
        """generate the child nodes and append them to nd's children"""
        stat = nd.state
        tempState = GameState(to_move=stat.to_move, move=stat.move, utility=stat.utility, board=stat.board,
                              moves=stat.moves)
        for a in self.game.actions(tempState):
            childNode = self.Node(self.game.result(tempState, a), nd)
            nd.children.append(childNode)

    def simulateRandomPlay(self, nd):
        currentState = nd.state
        while not self.isTerminalState(currentState.utility, currentState.moves):
            possibleMoves = self.game.actions(currentState)
            if not possibleMoves:
                break
            move = random.choice(possibleMoves)
            currentState = self.game.result(currentState, move)
        return self.game.to_move(currentState)

    def backPropagation(self, nd, winningPlayer):
        while nd is not None:
            nd.visitCount += 1
            if nd.state.to_move == winningPlayer:
                nd.winScore += 1
            nd = nd.parent
