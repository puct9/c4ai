#include "stdafx.h"
#include "MCTSEngine.h"
#include "MCTSNode.h"
#include <iostream>



MCTSEngine::MCTSEngine(C4Game position, Model * network, float c_puct, size_t playouts)
{
    this->nht = NodeHashtable(playouts * 8);  // make the thing 8x as many playouts

    this->top_node = this->nht.CreateNode(true);
    this->base_position = position;
    this->network = network;
    this->c_puct = c_puct;
    this->playouts = playouts;
}

MCTSEngine::MCTSEngine(C4Game position, Model * network, float c_puct, size_t playouts, size_t memory)
{
    this->nht = NodeHashtable(memory * 1024 * 1024 / sizeof(MCTSNode));

    this->top_node = this->nht.CreateNode(true);
    this->base_position = position;
    this->network = network;
    this->c_puct = c_puct;
    this->playouts = playouts;
}

void MCTSEngine::DoPlayouts()
{
    while (this->top_node->GetVisits() < this->playouts)
    {
        // find the appropriate leaf node
        C4Game look_position(base_position);  // copy it
        MCTSNode* leaf = this->top_node->ToLeaf(this->c_puct, look_position);

        if (leaf->IsTerminal())
        {
            leaf->Backprop(leaf->GetTerminalScore());
            continue;
        }

        this->network->SetPositionData(look_position.GetPositionArray());
        float** predictions = this->network->Run();

        leaf->Expand(look_position, predictions[1], this->nht);
        leaf->Backprop(-predictions[0][0]);
    }
}

void MCTSEngine::SetHashSizeByMemory(size_t megabytes)
{
    this->SetHashSizeByLength(megabytes * 1024 * 1024 / sizeof(MCTSNode));
}

void MCTSEngine::SetHashSizeByLength(size_t length)
{
    size_t* top_id_ = this->top_node->GetId();
    size_t top_id[2];  // copy
    top_id[0] = top_id_[0];
    top_id[1] = top_id_[1];
    int top_depth = this->top_node->GetDepth();
    this->nht.Rebuild(length);
    this->top_node = this->nht.GetNodeById(top_id, top_depth);
}

float * MCTSEngine::GetMoveProbs()
{
    this->DoPlayouts();

    for (int i = 0; i < 7; i++)
    {
        this->move_probs[i] = 0;
        MCTSNode* child = this->top_node->GetChild(i);
        if (child == nullptr)
            continue;
        this->move_probs[i] = (float)child->GetVisits() / (float)this->playouts;
    }

    return this->move_probs;
}

std::vector<int> MCTSEngine::GetPV()
{
    std::vector<int> pv;
    this->top_node->WriteInfoToPV(pv);
    return pv;
}

MCTSEngine::~MCTSEngine()
{
}
