#ifndef C4UCT_NODE_HASHTABLE_H
#define C4UCT_NODE_HASHTABLE_H

#include <vector>
#include "MCTSNode.h"


class NodeHashtable
{
private:
    std::vector<MCTSNode> storage;

public:
    NodeHashtable(size_t length = 10000);
    
    static MCTSNode* AddNode(MCTSNode& node, std::vector<MCTSNode>& arr);

    MCTSNode* AddNode(MCTSNode& node);
    MCTSNode* CreateNode(bool top);
    MCTSNode* CreateNode(MCTSNode *pa, int m, float p, bool t, int ts, size_t* identifier, int depth);
    MCTSNode* GetNodeById(size_t* id, int depth);
    size_t CountActive();
    void DestroyById(size_t* id, int depth);
    void Rebuild(size_t length);

    size_t ObjectGetDistance(size_t* id, int depth);

    void Show();

    size_t GetLength() { return this->storage.size(); };

    ~NodeHashtable();
};

#endif
