#ifndef C4UCT_MCTS_NODE_H
#define C4UCT_MCTS_NODE_H

#include "C4Game.h"
#include <vector>
#include <algorithm>


class NodeHashtable;


class MCTSNode
{
private:
    // basic tree traversal parameters
    bool terminal;
    float terminal_score;
    int move;

    // UCT params
    float P;
    size_t N;
    float W;

    // I mean, this is a tree, right?
    MCTSNode *children[7];
    MCTSNode *parent;

    // stuff
    size_t identifier[2];
    int depth;
    bool active;

public:

    MCTSNode(bool top = false);
    MCTSNode(MCTSNode *pa, int m, float p, bool t, int ts, size_t* id, int depth);

    void Expand(C4Game &state, float* priors, NodeHashtable& nht);
    void Backprop(float value);
    void RefreshChildrenPointers(NodeHashtable& nht);
    float Value(float c_puct);
    MCTSNode * ToLeaf(float c_puct, C4Game& position);
    void SetInactive();
    void SetParent(MCTSNode* new_parent);
    void ShowDetailedInfo();
    void WriteInfoToPV(std::vector<int>& pv);

    bool IsActive() { return this->active; };
    bool IsTerminal() { return this->terminal; };
    float GetTerminalScore() { return this->terminal_score; };
    int GetDepth() { return this->depth; };
    int GetMove() { return this->move; };
    size_t GetVisits() { return this->N; };
    size_t* GetId() { return this->identifier; };
    MCTSNode* GetChild(int n) { return this->children[n]; };
    float GetP() { return this->P; };
    float GetQ() { return this->N > 0 ? this->W / this->N : -1.0f; };

    ~MCTSNode();
};

#endif
