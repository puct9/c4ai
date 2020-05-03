#include <iostream>
#include <math.h>
#include "MCTSNode.h"
#include "NodeHashtable.h"



MCTSNode::MCTSNode(bool top)
    : parent(nullptr), move(0), P(0), N(0), W(0), terminal(false), terminal_score(0), active(top),
    depth(0)
{
    this->identifier[0] = 0;
    this->identifier[1] = 0;
#ifndef _WIN32
    for (int i = 0; i < 7; i++)
        this->children[i] = nullptr;
#endif
}

MCTSNode::MCTSNode(MCTSNode * pa, int m, float p, bool t, int ts, size_t* id, int depth)
    : parent(pa), move(m), P(p), N(0), W(0.0f), terminal(t), terminal_score((float)ts), active(true),
    depth(depth)
{
    // depth is the depth of the game at which the move m was played (lowest value 0)
    this->identifier[0] = id[0];
    this->identifier[1] = id[1];
    // depth increment
    if (depth < 21)
        this->identifier[0] += pow(7, depth) * m;
    else
        this->identifier[1] += pow(7, depth - 21) * m;
#ifndef _WIN32
    for (int i = 0; i < 7; i++)
        this->children[i] = nullptr;
#endif
}

void MCTSNode::Expand(C4Game & state, float * priors, NodeHashtable& nht)
{
    bool* legal = state.LegalMoves();
    float legalsum = 0;
    for (int i = 0; i < 7; i++)
    {
        if (legal[i])
        {
            legalsum += priors[i];
        }
    }
    for (int i = 0; i < 7; i++)
    {
        if (legal[i])
        {
            // check terminal
            state.PlayMove(i);
            float state_res = (float)state.GameOver();
            children[i] = nht.CreateNode(this, i, priors[i] / legalsum, state_res > -1, state_res, this->identifier,
                this->depth + 1);
            state.UndoMove();
        }
    }
}

void MCTSNode::Backprop(float value)
{
    this->N++;
    this->W += value;
    if (this->parent != nullptr)
        this->parent->Backprop(-value);
}

void MCTSNode::RefreshChildrenPointers(NodeHashtable & nht)
{
    // update the array
    for (int i = 0; i < 7; i++)
    {
        // compute the next id
        size_t predicted_id[2];
        predicted_id[0] = this->identifier[0];
        predicted_id[1] = this->identifier[1];
        // depth increment
        if (this->depth + 1 < 21)
            predicted_id[0] += pow(7, this->depth + 1) * i;
        else
            predicted_id[1] += pow(7, this->depth + 1 - 21) * i;

        this->children[i] = nht.GetNodeById(predicted_id, this->depth + 1);
    }
    // recursively propagate downwards
    for (auto& c : this->children)
    {
        if (c == nullptr)
            continue;
        c->SetParent(this);  // I am your father
        c->RefreshChildrenPointers(nht);
    }
}

float MCTSNode::Value(float c_puct)
{
    if (terminal && terminal_score != 0)
        return 999;  // arbitrarily large, impossible value
    float u = (log((parent->N + 19653.0f) / 19652.0f) + c_puct) * P * pow(parent->N, 0.5f) / (1.0f + N);
    float q = N == 0 ? -1 : W / N;  // FPU -1 with 0 playouts
    return q + u;
}

MCTSNode * MCTSNode::ToLeaf(float c_puct, C4Game& position)
{
    int best_i = -1;
    float best_value = -999;  // arbitrarily small, mathematically impossible value
    for (int i = 0; i < 7; i++)
    {
        if (children[i] == nullptr)
            continue;
        float child_value = children[i]->Value(c_puct);
        if (child_value > best_value)
        {
            best_value = child_value;
            best_i = i;
        }
    }
    if (best_i == -1)
        return this;
    position.PlayMove(best_i);
    return children[best_i]->ToLeaf(c_puct, position);
}

void MCTSNode::SetInactive()
{
    this->active = false;
    for (auto& child : this->children)
    {
        if (child == nullptr)
            continue;
        child->SetInactive();
    }
}

void MCTSNode::SetParent(MCTSNode * new_parent)
{
    this->parent = new_parent;
}

void MCTSNode::ShowDetailedInfo()
{
    std::cout << "Node ID: " << this->identifier[0] << " " << this->identifier[1] << "\nDepth: " <<
        this->depth << "\nN = " << this->N << "\nP = " << this->P << "\nW = " << this->W <<
        std::endl;
}

void MCTSNode::WriteInfoToPV(std::vector<int>& pv)
{
    // get the child with highest N
    size_t best_n = 0;
    MCTSNode* best_c = nullptr;
    for (auto& c : this->children)
    {
        if (c == nullptr)
            continue;
        if (c->GetVisits() > best_n)
        {
            best_n = c->GetVisits();
            best_c = c;
        }
    }
    if (best_c != nullptr)
    {
        pv.push_back(best_c->GetMove());
        best_c->WriteInfoToPV(pv);
    }
}


MCTSNode::~MCTSNode()
{
}
