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

        depth = 1
        bestPossibleMove = None

        """Use timer above to apply iterative deepening"""
        while time.perf_counter() < end:
            self.root.visitCount = 0
            self.runMCTS(end, depth)
            depth += 1
            bestPossibleMove = self.root.getChildWithMaxScore().state.move
            print(f"Best move at depth {depth - 1}: {bestPossibleMove}")

        actualTimeSpent = time.perf_counter() - start
        print(f"Time allotted: {timelimit}s, Time spent: {actualTimeSpent:.2f}s, Depth reached: {depth - 1}")
        return bestPossibleMove

    def runMCTS(self, end, depth):
        while time.perf_counter() < end:
            # count = 100  # use this and the next line for debugging. Just disable previous while and enable these 2 lines
            # while count >= 0:
            # count -= 1

            # SELECT stage use selectNode()
            nodeForExploration = self.selectNode(self.root, depth)
            print(f"Selected node for exploration: {nodeForExploration.state.move}")

            # EXPAND stage
            self.expandNode(nodeForExploration)
            print(
                f"Expanded node: {nodeForExploration.state.move} with children: {[child.state.move for child in nodeForExploration.children]}")

            # SIMULATE stage using simuplateRandomPlay()
            simulationResult = self.simulateRandomPlay(nodeForExploration)
            print(f"Simulation result: {simulationResult}")

            # BACKUP stage using backPropagation
            self.backPropagation(nodeForExploration, simulationResult)
            print(f"Backpropagation completed for node: {nodeForExploration.state.move}")

    """selection stage function. walks down the tree using findBestNodeWithUCT()"""

    def selectNode(self, nd, depth):
        node = nd
        currentDepth = 0
        while not self.isTerminalState(node.state.utility, node.state.moves) and currentDepth < depth:
            if len(node.children) == 0:
                self.expandNode(node)
            else:
                node = self.findBestNodeWithUCT(node)
            currentDepth += 1
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

    def simulateRandomPlay(self, node):
        isItWinner = self.game.compute_utility(node.state.board, node.state.move, node.state.board[node.state.move])
        if isItWinner == self.game.k:
            assert (node.state.board[node.state.move] == 'X')
            if node.parent is not None:
                node.parent.winScore = -sys.maxsize
            return 'X' if isItWinner > 0 else 'O'

        currentState = copy.deepcopy(node.state)

        while not self.isTerminalState(currentState.utility, currentState.moves):
            move = random.choice(currentState.moves)
            currentState = self.game.result(currentState, move)

        isItWinner = self.game.compute_utility(currentState.board, currentState.move,
                                               currentState.board[currentState.move])

        return 'X' if isItWinner > 0 else 'O' if isItWinner < 0 else 'N'

    def backPropagation(self, nd, winningPlayer):
        tempNode = nd
        while tempNode is not None:
            tempNode.visitCount += 1
            if tempNode.state.to_move != winningPlayer:
                tempNode.winScore += 1
            tempNode = tempNode.parent
