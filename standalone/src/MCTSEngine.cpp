#include <iostream>
#include "MCTSEngine.h"
#include "MCTSNode.h"



MCTSEngine::MCTSEngine(C4Game position, Model * network, float c_puct, size_t playouts)
{
    // make the thing 8x as many playouts
    this->nht = NodeHashtable(playouts * 8 + 1);  // adding 1 makes 4head into 5head
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

void MCTSEngine::DoPlayouts(bool verbose)
{
    std::vector<int> prevpv;
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

        if (verbose && this->playouts % 50 == 0)
        {
            std::vector<int> currpv = this->GetPV();
            bool changed = prevpv.size() != currpv.size();
            if (!changed)
            {
                for (int i = 0; i < currpv.size(); i++)
                {
                    if (prevpv[i] != currpv[i])
                    {
                        changed = true;
                        break;
                    }
                }
            }
            if (changed)
            {
                std::cout << "info pv ";
                for (auto& move : currpv)
                    std::cout << move << ' ';
                std::cout << std::endl;
                prevpv = currpv;
            }
        }
    }

    if (verbose)
    {
        for (int i = 0; i < 7; i++)
        {
            MCTSNode* child = top_node->GetChild(i);
            if (child == nullptr)
                continue;
            std::cout << "[NODE] move " << i << " N " << child->GetVisits() <<
                " P " << child->GetP() << " Q " << child->GetQ() << '\n';
        }
        std::cout << "endinfo" << std::endl;
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
        this->move_probs[i] = (float)child->GetVisits() / ((float)this->playouts - 1);
    }

    return this->move_probs;
}

std::vector<int> MCTSEngine::GetPV()
{
    std::vector<int> pv;
    this->top_node->WriteInfoToPV(pv);
    return pv;
}

void MCTSEngine::RecycleTree(int move)
{
    // set the others to inactive
    for (int i = 0; i < 7; i++)
    {
        if (i != move && top_node->GetChild(i) != nullptr)
            top_node->GetChild(i)->SetInactive();
    }
    size_t new_top_node_id[2];  // the old object holding this is going to die so we copy
    int new_top_node_depth = top_node->GetChild(move)->GetDepth();
    auto new_top_node_id_ref = top_node->GetChild(move)->GetId();
    new_top_node_id[0] = new_top_node_id_ref[0]; new_top_node_id[1] = new_top_node_id_ref[1];
    top_node->SetOnlyThisNodeAsInactive();
    this->nht.Rebuild(this->nht.GetLength());
    base_position.PlayMove(move);
    this->top_node = this->nht.GetNodeById(new_top_node_id, new_top_node_depth);
    this->top_node->SetAsTopNode();
}

MCTSEngine::~MCTSEngine()
{
}
