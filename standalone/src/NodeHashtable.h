#ifndef C4UCT_NODE_HASHTABLE_H
#define C4UCT_NODE_HASHTABLE_H

#include <vector>
#include "MCTSNode.h"


class NodeHashtable
{
private:
    std::vector<MCTSNode> storage;

public:
    NodeHashtable(std::uint64_t length = 10000);
    
    static MCTSNode* AddNode(MCTSNode& node, std::vector<MCTSNode>& arr);

    MCTSNode* AddNode(MCTSNode& node);
    MCTSNode* CreateNode(bool top);
    MCTSNode* CreateNode(MCTSNode *pa, int m, float p, bool t, int ts, std::uint64_t* identifier, int depth);
    MCTSNode* GetNodeById(std::uint64_t* id, int depth);
    std::uint64_t CountActive();
    void DestroyById(std::uint64_t* id, int depth);
    void Rebuild(std::uint64_t length);

    std::uint64_t ObjectGetDistance(std::uint64_t* id, int depth);

    void Show();

    std::uint64_t GetLength() { return this->storage.size(); };

    ~NodeHashtable();
};

#endif
