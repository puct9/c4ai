#pragma once
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

    void DoPlayouts();
    void SetHashSizeByMemory(size_t megabytes);
    void SetHashSizeByLength(size_t length);
    float* GetMoveProbs();
    std::vector<int> GetPV();

    MCTSNode* GetTopNode() { return this->top_node; };

    ~MCTSEngine();
};

