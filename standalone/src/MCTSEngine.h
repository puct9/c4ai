#ifndef C4UCT_MCTS_ENGINE_H
#define C4UCT_MCTS_ENGINE_H
constexpr auto C4UCT_MCTS_ENGINE_INFO_VOMIT = false;


#include "C4Game.h"
#include "Model.h"
#include "MCTSNode.h"
#include "NodeHashtable.h"


class MCTSEngine
{
private:
    NodeHashtable nht;

    MCTSNode* top_node;
    C4Game base_position;
    Model* network;
    float c_puct;
    size_t playouts;

    float move_probs[7];

public:
    MCTSEngine(C4Game position, Model* network, float c_puct, size_t playouts);
    MCTSEngine(C4Game position, Model* network, float c_puct, size_t playouts, size_t memory);

    void DoPlayouts(bool verbose = false);
    void SetHashSizeByMemory(size_t megabytes);
    void SetHashSizeByLength(size_t length);
    float* GetMoveProbs();
    std::vector<int> GetPV();
    void RecycleTree(int move);

    void PeekHT() { this->nht.Show(); };
    size_t PeekHTCapacity() { return this->nht.CountActive(); };
    MCTSNode* GetTopNode() { return this->top_node; };

    ~MCTSEngine();
};

#endif
