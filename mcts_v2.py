"""
Version 2 of UCT (MCTS variant) search engine
The main differences:
    - C_PUCT now grows with search time, as per A0 paper
    - Playouts are now 'batched'. Whilst playouts are now slightly less
        quality, the overall search should be faster and stronger
    - Always select forced win to avoid issues with virtual loss (see batching)
        and the 'U' term in computing the score for a node in greedy node
        selection in finding the leaf node in the search
    - Forced losses are pruned (Q=-inf); this is supposed to help especially
        at low node counts
    - FPU (first play urgency) set to -1, as per A0 paper. This means that
        nodes without any visits will return a default Q value of -1, assuming
        that the move is losing
"""
import random
import time
from typing import List

import numpy as np
from keras.models import Model

from c4game import C4Game


USE_ITERATIVE_FOR_GREEDY_TRAVERSAL = False
DO_SEARCH_TREE_PRUNING = False


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# the above and below are the same (shown algebraically)
# the above is better as it avoids overflow
# the bottom line is L1(V - k1) = L1(V - k2) for all k1, k2 in R


def l1_exp(x):
    probs = np.exp(x)
    probs /= np.sum(probs)
    return probs


class MCTSNode:
    """
    A node of MCTS tree
    """

    # memory efficiency and performance
    __slots__ = ('move', 'parent', 'children', 'prune', 'terminal',
                 'terminal_score', 'P', 'N', 'W', 'VL')

    def __init__(self, parent: "MCTSNode" = None,
                 move: int = None, prior: float = None,
                 terminal: bool = False, terminal_score: int = 0) -> None:
        """
        Parameters
        ----------
        parent: `MCTSNode`
            The parent node
        move: `int`
            The move which was played to reach this node
        prior: `float`
            The probability the policy assigns this move
        terminal: `bool`
            True if this node is a game end state, False if not
        terminal_score: `int`
            The score for the node given it is a terminal position
        """
        self.move = move
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.prune = False  # set to true if it is a losing move
        self.terminal = terminal  # denotes the winner of the game.
        self.terminal_score = terminal_score  # 1 if win, 0 if tie
        self.P = prior  # prior probability of selecting this move
        self.N = 0  # number of visits; default value
        self.W = 0  # cumulative of value backpropagation; default value
        self.VL = 0  # virtual loss; default value

    @property
    def Q(self) -> int:
        if self.prune:
            return -2 * self.N + ((self.W - self.VL) /
                                  (self.N + self.VL))
        if not self.N + self.VL:
            return -1  # FPU in alphazero is -1
        # do not apply VL if terminal
        return ((self.W - self.VL) /
                (self.N + self.VL))

    def value(self, c_puct: float) -> float:
        """
        Returns the value for greedy selection for tree traversal to a leaf
        node
        Parameters
        ----------
        c_puct: `float`
            Constant controlling exploration
        Returns
        -------
        score: `float`
            (Q + log((parent_N + cb) / cb) + cp) * P * sqrt(parent_N) / (1 + N)
        """
        if self.terminal and self.terminal_score:  # win
            return float('inf')
        # c_puct_base = 19652, as described in alphazero
        scale = np.log((self.parent.N + 19652 + 1) / 19652) + c_puct
        u = (scale * self.P * (self.parent.N  # + self.parent.VL - self.VL
                               ) ** 0.5 /
             (1 + self.N))  # + self.VL?
        return self.Q + u

    def backprop(self, value: float) -> None:
        """
        Backproagates a value from leaf node to top node
        Parameters
        ----------
        value: `float`
            The value to backpropagate up the search tree.
            Leaf node W will be updated with value
        """
        # first we update ourselves
        self.N += 1
        self.W += value
        # now we backprop
        # remove our virtual loss
        self.VL -= 1
        if self.parent:  # is not None
            self.parent.backprop(-value)

    def to_leaf(self, c_puct: float, position: C4Game) -> 'MCTSNode':
        """
        Traverses the tree from current node to a leaf node
        Parameters
        ----------
        c_puct: `float`
            Constant controlling exploration
        position: `C4Game`
            The position of the current game state, which will be automatically
            updated as the tree is traversed to a leaf node.
        Returns
        -------
        leaf_node: `MCTSNode`
            The leaf node found after tree traversal
        """
        if USE_ITERATIVE_FOR_GREEDY_TRAVERSAL:
            curr = self
            while True:
                curr.VL += 1
                if curr.move is not None:
                    position.play_move(curr.move)
                if not curr.children or curr.terminal:
                    return curr
                child_scores = [c.value(c_puct) for c in curr.children]
                curr = curr.children[child_scores.index(max(child_scores))]
            raise Exception("Tree traversal error")
        # BELOW: RECURSIVE ALGORITHM
        self.VL += 1
        if self.move is not None:
            position.play_move(self.move)
        if not self.children:
            return self
        # more performant than np.argmax by a lot
        # select the best child
        max_child_score = float('-inf')
        max_child_index = 0
        for i, c in enumerate(self.children):
            v = c.value(c_puct)
            if v > max_child_score:
                max_child_score = v
                max_child_index = i
        if DO_SEARCH_TREE_PRUNING:
            if max_child_score < -1:  # all losing, this move is won
                self.terminal = True
                self.terminal_score = 1
                return self
            elif max_child_score == float('inf'):  # this move is lost
                self.prune = True  # this node will never be selected again
        return (self.children[max_child_index].to_leaf(c_puct, position))

    def expand(self, priors: np.ndarray, position: C4Game) -> None:
        """
        Adds children to the current node
        Parameters
        ----------
        priors:
            A vector of the NN's prior probabilities for each child in order
        position:
            The game state required to reach this node
        """
        allowed = position.legal_moves()
        for (mv, prior) in enumerate(priors):
            move = mv
            if allowed[move]:
                # empty square
                position.play_move(move)
                term = position.check_terminal()
                is_term = term is not None
                self.children.append(MCTSNode(self, move, prior, is_term,
                                              term if term is not None else 0))
                position.undo_move()

    def walk_pv(self) -> 'MCTSNode':
        """
        Traverses the current tree using node count instead of value
        Returns
        -------
        end_node: `MCTSNode`
            The end of the principal variation
        """
        best_n = 0
        best_c = None
        for c in self.children:
            if c.N > best_n:
                best_n = c.N
                best_c = c
        if best_c is None:
            return self
        return best_c.walk_pv()

    def __str__(self) -> str:
        return f'[NODE] MV={self.move} N={self.N} Q={self.Q} P={self.P}'


class MCTS:
    """
    MCTS search system
    """

    def __init__(self, position: C4Game, stochastic: bool, network: Model,
                 c_puct: float, playouts: int, batch_size: int = 16,
                 dir_alpha: float = 1.4):
        """
        Parameters
        ----------
        position: `C4Game`
            The current game position to search from
        stochastic: `bool`
            Set to true if this is a selfplay training game
        network: `keras.models.Model`
            The neural network
        c_puct: `float`
            Constant controlling exploration
        playouts: `int`
            Number of playouts to make for search
        dir_alpha: `float`
            Diriclet alpha for selfplay training games, defaults to 1.4
        """
        # team is -1 for black to play, 1 for white to play
        self.top_node = MCTSNode()
        self.base_position = position  # shallow naming
        self.network = network
        self.c_puct = c_puct
        self.playouts = playouts
        self.stochastic = stochastic
        self.dir_alpha = dir_alpha
        self.batch_size = batch_size  # for parallel-ish

    def playout_to_max(self) -> np.ndarray:
        """
        Returns
        -------
        search_probs: `np.ndarray`
            A vector of move probabilites following mcts
        """
        while self.top_node.N < self.playouts:
            # recursively greedily select node via puct algorithm
            leafs = []  # parallel-ish
            for _ in range(self.batch_size):
                look_position = self.base_position.state_copy()
                leaf = self.top_node.to_leaf(self.c_puct, look_position)
                leafs.append((leaf, look_position))

            # evaluate
            evaluations = [None] * self.batch_size
            batch_priors = [None] * self.batch_size
            batch_positions = []
            for i, (leaf, look_position) in enumerate(leafs):
                if leaf.terminal:
                    evaluations[i] = -abs(leaf.terminal_score)
                else:
                    batch_positions.append(look_position.state)

            # use the neural network
            # 50% chance to flip every position in the batch in training
            flipped = False
            if self.stochastic and random.random() > 0.5:
                flipped = True
                batch_positions = batch_positions[:, ::-1, :, :]
            if batch_positions:
                leaf_value, priors = self.network.predict(
                    np.array(batch_positions))
                if flipped:
                    priors = priors[:, ::-1]
            else:
                leaf_value, priors = [], []
            # populate evaluations
            _count = 0
            for i, ev in enumerate(evaluations):
                if ev is None:
                    evaluations[i] = leaf_value[_count, 0]
                    # if we needed an evaluation we also need an expansion
                    batch_priors[i] = priors[_count]
                    _count += 1
            # leaf_value is how good it is for CURRENT player of the state
            for (leaf, look_position), prior in zip(leafs, batch_priors):
                # we could have 2 or more searches on one leaf
                if not leaf.children and not leaf.terminal:
                    leaf.expand(prior, look_position)
            # backprop
            for (leaf, _), ev in zip(leafs, evaluations):
                leaf.backprop(-ev)

        # calculate root children probabilities, and fill in the invalid ones
        # with 0
        root_children_probs = []
        ind = 0
        allowed = self.base_position.legal_moves()
        for mv in range(7):
            move = mv
            if allowed[move]:
                root_children_probs.append(self.top_node.children[ind].N /
                                           self.top_node.N)
                ind += 1
            else:
                root_children_probs.append(0)

        return root_children_probs

    def search_for_time(self, duration: float) -> np.ndarray:
        """
        Parameters
        ----------
        duration: `float`
            The duration of time to search the position
        Returns
        -------
        search_probs: `np.ndarray`
            A vector of move probabilites following mcts
        """
        start_time = time.time()
        ret = self.playout_to_max()
        while time.time() - start_time < duration:
            self.playouts += self.batch_size * 3
            ret = self.playout_to_max()
        return ret

    def pick_move(self, temp: float = 1e-3
                  ) -> int:
        """
        Parameters
        ----------
        temp: `float`
            Temperature for selfplay training games
        Returns
        -------
        move: `int`
            The move to make in the position
        """
        search_probs = self.playout_to_max()
        if not max(search_probs):  # == 0:
            raise ValueError('Wtf')
        if not self.stochastic:
            ind = search_probs.index(max(search_probs))
            return ind
        # stochastic = selfplay game
        # apply the dirichlet noise at move selection
        dirichlet = np.random.dirichlet([self.dir_alpha] * len(search_probs))
        noisy_probs = np.array(search_probs) * 0.84 + dirichlet * 0.16
        # normally we would
        # let v = a vector of visits
        # v ^ (1 / temp)
        # but due to overflow problems, we rearrange
        # v ^ (1 / temp) = exp(log(v ^ (1 / temp))) = exp(log(v) / temp)
        new_probs = softmax(np.log(noisy_probs + 1e-10) / temp)
        return np.random.choice(self.top_node.children, 1, p=new_probs)[0].move

    def get_pv(self) -> List[MCTSNode]:
        """
        Returns the principal variation
        Returns
        -------
        full_pv: `List[MCTSNode]`
            The principal variation from first move to end from greedy N count
            tree traversal
        """
        end_pv = self.top_node.walk_pv()
        full_pv = []
        while end_pv.parent is not None:
            full_pv.append(end_pv)
            end_pv = end_pv.parent
        return full_pv[::-1]
