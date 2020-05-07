#include <algorithm>
#include <cfloat>
#include <math.h>
#include <iostream>
#include <sstream>
#include "DebugMode.h"
#include "C4Game.h"
#include "MCTSEngine.h"
#include "ModelManager.h"


int DBGMode()
{
    ModelManager model_manager;
    std::cout.flush();
#ifdef _WIN32
    Model* model = model_manager.CreateModel(L"Models/default.onnx");
#else
    Model* model = model_manager.CreateModel("Models/default.onnx");
#endif
    C4Game game;

    std::cout << "Welcome to debug mode." << std::endl;

    // debug mode has a persistent MCTSEngine
    MCTSEngine engine = MCTSEngine(game, model, 3, 800);

    // input loop
    std::string user_in;
    unsigned long user_value;
    for (;;)
    {
        std::getline(std::cin, user_in);
        if (user_in.size() == 0)
            continue;
        if (user_in == "isready")
            std::cout << "readyok" << std::endl;
        if (user_in == "peek")
            engine.PeekHT();
        if (user_in == "go")
            engine.DoPlayouts();
        if (user_in == "cap")
            std::cout << engine.PeekHTCapacity() << std::endl;
        if (user_in.rfind("prune ", 0) == 0)
        {
            std::string number = user_in.substr(6);
            std::stringstream(number) >> user_value;
            engine.GetTopNode()->GetChild(user_value)->SetInactive();
        }
        if (user_in.rfind("select ", 0) == 0)
        {
            std::string number = user_in.substr(7);
            std::stringstream(number) >> user_value;
            engine.RecycleTree(user_value);
        }
        if (user_in == "exit")
            return -1;
    }
    return 0;
}
