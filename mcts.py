from typing import List

import numpy as np
from keras.models import Model

from c4game import C4Game


USE_ITERATIVE_FOR_GREEDY_TRAVERSAL = False


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
        self.terminal = terminal  # denotes the winner of the game.
        self.terminal_score = terminal_score  # 1 if win, -1 if lose, 0 if tie
        self.P = prior  # prior probability of selecting this move
        self.N = 0  # number of visits
        self.Q = 0  # default value
        self.W = 0  # default value

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
            Q + c_puct * P * sqrt(parent_N) / (1 + N)
        """
        u = (c_puct * self.P * (self.parent.N) ** 0.5 / (1 + self.N))
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
        self.Q = self.W / self.N
        # now we backprop
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
                if curr.move is not None:
                    position.play_move(curr.move)
                if not curr.children:
                    return curr
                child_scores = [c.value(c_puct) for c in curr.children]
                curr = curr.children[child_scores.index(max(child_scores))]
            raise Exception("Tree traversal error")
        # BELOW: OLD RECURSIVE ALGORITHM
        if self.move is not None:  # move is (None, None) if passing
            position.play_move(self.move)
        if not self.children:
            return self
        # more performant than np.argmax by a lot
        # select the best child
        child_scores = [c.value(c_puct) for c in self.children]
        return (self.children[child_scores.index(max(child_scores))]
                .to_leaf(c_puct, position))

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
                 c_puct: float, playouts: int, dir_alpha: float = 1.4):
        """
        Parameters
        ----------
        position: `C4Game`
            The current game position to search from
        stochastic: `bool`
            Set to true if this is a selfplay training game
        network: `keras.models.Model`
            The neural netowrk
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

    def playout_to_max(self, dir_alpha: float = 1.4) -> np.ndarray:
        """
        Returns
        -------
        search_probs: `np.ndarray`
            A vector of move probabilites following mcts
        """
        do_iters = self.playouts - self.top_node.N  # must not change
        done_iters = 0
        while done_iters < do_iters:
            done_iters += 1
            # recursively greedily select node via puct algorithm
            look_position = self.base_position.state_copy()
            leaf = self.top_node.to_leaf(self.c_puct, look_position)

            # evaluate
            if leaf.terminal:
                # backprop also undoes all our moves
                leaf.backprop(abs(leaf.terminal_score))
                continue

            # use the neural network
            leaf_value, priors = self.network.predict(
                np.array([look_position.state]))
            # leaf_value is how good it is for CURRENT player of the state
            # expand, but before that, add dirichlet noise
            # dirichlet noise for legal moves only
            if self.stochastic:  # we are doing a selfplay game
                legal_moves = look_position.legal_moves()
                _dirichlet = np.random.dirichlet(
                    [self.dir_alpha] * sum(legal_moves))
                dirichlet = np.zeros(7)
                ind = 0
                for i, v in enumerate(legal_moves):
                    if v:
                        dirichlet[i] = _dirichlet[ind]
                        ind += 1
                leaf.expand((priors[0] + dirichlet) * 0.5, look_position)
            else:
                leaf.expand(priors[0], look_position)
            if leaf.move is None:  # toppest node, first playout
                leaf.backprop(-leaf_value[0, 0])
                continue
            # backprop
            leaf.backprop(-leaf_value[0, 0])

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
        visits = [n.N for n in self.top_node.children]
        # normally we would
        # let v = a vector of visits
        # v ^ (1 / temp)
        # but due to overflow problems, we rearrange
        # v ^ (1 / temp) = exp(log(v ^ (1 / temp))) = exp(log(v) / temp)
        new_probs = softmax(np.log(np.array(visits) + 1e-10) / temp)
        # give this a try
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
