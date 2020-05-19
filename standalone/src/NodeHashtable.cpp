#include "NodeHashtable.h"
#include "MCTSNode.h"
#include <iostream>



NodeHashtable::NodeHashtable(std::uint64_t length)
{
    /*
    std::cout << "Creating hashtable with size: " << length << std::endl;
    std::cout << "Memory usage: " << length * sizeof(MCTSNode) / (float)(1024 * 1024) << 
        " MB" << std::endl;
    */
    this->storage = std::vector<MCTSNode>(length);
}

MCTSNode * NodeHashtable::AddNode(MCTSNode & node)
{
    std::uint64_t* id = node.GetId();
    std::uint64_t pos = (id[0] + id[1]) % this->storage.size();
    for (std::uint64_t offset = 0; offset < this->storage.size(); offset++)
    {
        if (!this->storage[(pos + offset) % this->storage.size()].IsActive())
        {
            pos = (pos + offset) % this->storage.size();
            break;
        }
    }
    // don't check for no more room as it won't happen
    this->storage[pos] = node;  // this is a copy
    return &this->storage[pos];
}

MCTSNode * NodeHashtable::AddNode(MCTSNode & node, std::vector<MCTSNode>& arr)
{
    std::uint64_t* id = node.GetId();
    std::uint64_t pos = (id[0] + id[1]) % arr.size();
    for (std::uint64_t offset = 0; offset < arr.size(); offset++)
    {
        if (!arr[(pos + offset) % arr.size()].IsActive())
        {
            pos = (pos + offset) % arr.size();
            break;
        }
    }
    // don't check for no more room as it won't happen
    arr[pos] = node;  // this is a copy
    return &arr[pos];
}

MCTSNode * NodeHashtable::CreateNode(bool top)
{
    // "hash"
    // because math, it won't overflow, and even if it does, it's not a big deal...
    MCTSNode new_node = MCTSNode(top);
    std::uint64_t* new_id = new_node.GetId();
    std::uint64_t pos = (new_id[0] + new_id[1]) % this->storage.size();
    // get pos
    for (std::uint64_t offset = 0; offset < this->storage.size(); offset++)
    {
        if (!this->storage[(pos + offset) % this->storage.size()].IsActive())
        {
            pos = (pos + offset) % this->storage.size();
            break;
        }
    }
    if (this->storage[pos].IsActive())
    {
        // hashtable full

    }
    this->storage[pos] = new_node;
    return &this->storage[pos];
}

MCTSNode * NodeHashtable::CreateNode(MCTSNode * pa, int m, float p, bool t, int ts, std::uint64_t * identifier,
    int depth)
{
    // "hash"
    // because math, it won't overflow, and even if it does, it's not a big deal...
    MCTSNode new_node = MCTSNode(pa, m, p, t, ts, identifier, depth);
    std::uint64_t* new_id = new_node.GetId();
    std::uint64_t pos = (new_id[0] + new_id[1]) % this->storage.size();
    // get pos
    for (std::uint64_t offset = 0; offset < this->storage.size(); offset++)
    {
        if (!this->storage[(pos + offset) % this->storage.size()].IsActive())
        {
            pos = (pos + offset) % this->storage.size();
            break;
        }
    }
    if (this->storage[pos].IsActive())
    {
        // hashtable full
        std::cout << "CRITICAL ERROR HASHTABLE OVERFULL, FAILED TO CREATE NODE" << std::endl;
        return nullptr;
    }
    this->storage[pos] = new_node;
    return &this->storage[pos];
}

MCTSNode * NodeHashtable::GetNodeById(std::uint64_t * id, int depth)
{
    std::uint64_t pos = (id[0] + id[1]) % this->storage.size();
    for (std::uint64_t offset = 0; offset < this->storage.size(); offset++)
    {
        if (!this->storage[(pos + offset) % this->storage.size()].IsActive())
            return nullptr;  // node doesn't exist
        std::uint64_t* local_id = this->storage[(pos + offset) % this->storage.size()].GetId();
        if (local_id[0] == id[0] && local_id[1] == id[1] &&
            this->storage[(pos + offset) % this->storage.size()].GetDepth() == depth)
        {
            pos = (offset + pos) % this->storage.size();
            break;
        }
    }
    std::uint64_t* found_id = this->storage[pos].GetId();
    if (found_id[0] == id[0] && found_id[1] == id[1] && this->storage[pos].GetDepth() == depth)
        return &this->storage[pos];
    return nullptr;  // node doesn't exist
}

std::uint64_t NodeHashtable::CountActive()
{
    std::uint64_t counter = 0;
    for (std::uint64_t i = 0; i < this->storage.size(); i++)
    {
        if (this->storage[i].IsActive())
            counter++;
    }
    return counter;
}

void NodeHashtable::DestroyById(std::uint64_t * id, int depth)
{
    this->GetNodeById(id, depth)->SetInactive();
}

void NodeHashtable::Rebuild(std::uint64_t length)
{
    std::vector<MCTSNode> new_ht = std::vector<MCTSNode>(length);
    int min_depth = 69420;  // impossibly high depth
    std::uint64_t min_depth_node_id[2];
    for (std::uint64_t i = 0; i < this->storage.size(); i++)
    {
        auto& elem = this->storage[i];
        if (elem.IsActive())
        {
            auto new_node = NodeHashtable::AddNode(this->storage[i], new_ht);
            if (elem.GetDepth() < min_depth)
            {
                min_depth = elem.GetDepth();
                auto eid = elem.GetId();
                min_depth_node_id[0] = eid[0]; min_depth_node_id[1] = eid[1];
            }
        }
    }
    this->storage = new_ht;
    // there is a bug which stops the below from working for some reason!
    auto top_node = this->GetNodeById(min_depth_node_id, min_depth);
    if (top_node->IsActive())
        top_node->RefreshChildrenPointers(*this);
}

std::uint64_t NodeHashtable::ObjectGetDistance(std::uint64_t * id, int depth)
{
    std::uint64_t pos = (id[0] + id[1]) % this->storage.size();
    for (std::uint64_t offset = 0; offset < this->storage.size(); offset++)
    {
        std::uint64_t* local_id = this->storage[(pos + offset) % this->storage.size()].GetId();
        if (local_id[0] == id[0] && local_id[1] == id[1] &&
            this->storage[(pos + offset) % this->storage.size()].GetDepth() == depth)
        {
            pos = (offset + pos) % this->storage.size();
            break;
        }
    }
    std::uint64_t* found_id = this->storage[pos].GetId();
    if (found_id[0] == id[0] && found_id[1] == id[1] && this->storage[pos].GetDepth() == depth)
    {
        std::uint64_t pos_1 = (id[0] + id[1]) % this->storage.size();
        return pos_1 > pos ? this->storage.size() - (pos_1 - pos) : pos - pos_1;
    }
    return 66666;  // spooky
}

void NodeHashtable::Show()
{
    std::cout << "----------------" << std::endl;
    for (std::uint64_t i = 0; i < this->storage.size(); i++)
    {
        std::cout << i << " | ";
        if (!this->storage[i].IsActive())
            std::cout << "INACTIVE" << std::endl;
        else
        {
            auto& node = this->storage[i];
            std::cout << node.GetId()[0] << " " <<
                node.GetId()[1] << " " << node.GetDepth() <<
                " P=" << node.GetP() << std::endl;
        }
    }
    std::cout << "----------------" << std::endl;
}

NodeHashtable::~NodeHashtable()
{
}
