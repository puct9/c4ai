#include <iostream>
#include <string>
#include <sstream>
#include "gsl/gsl_rng.h"
#include "AnalysisMode.h"
#include "MCTSEngine.h"
#include "MCTSNode.h"
#include "Model.h"
#include "ModelManager.h"
#include "TrainUtils.h"


int AnalysisMode()
{
    ModelManager model_manager;
    std::cout.flush();

    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);

    Model* model = model_manager.CreateModel(L"Models/save_10k.onnx");
    C4Game game;

    std::cout << "Welcome to analysis mode." << std::endl;

    // input loop
    std::string user_in;
    unsigned long user_value;
    for (;;)
    {
        std::getline(std::cin, user_in);
        if (user_in.size() == 0)
            continue;
        if (user_in.rfind("mv ", 0) == 0)
        {
            std::string number = user_in.substr(3, 1);
            std::stringstream(number) >> user_value;
            if (game.LegalMoves()[user_value])
                game.PlayMove(user_value);
        }
        if (user_in == "d")
        {
            game.Show();
        }
        if (user_in == "game")
            return 2;  // enter GameMode
        if (user_in == "isready")
            std::cout << "readyok" << std::endl;
        if (user_in.rfind("getbest n ", 0) == 0)
        {
            std::string number = user_in.substr(10);
            std::stringstream(number) >> user_value;
            if (user_value < 10)
                user_value = 10;
            MCTSEngine eng = MCTSEngine(game, model, 3, user_value);
            eng.DoPlayouts();
            auto pv = eng.GetPV();
            if (pv.size() == 0)
                std::cout << "end of game" << std::endl;
            std::cout << eng.GetTopNode()->GetChild(eng.GetPV()[0])->GetQ()
                << " " << eng.GetPV()[0] << std::endl;
        }
        if (user_in == "undo")
            game.UndoMove();
        if (user_in.rfind("position set ", 0) == 0)
        {
            std::string posstr = user_in.substr(13);
            game = C4Game(posstr);
        }
        if (user_in == "ssp")
            return 1;  // enter SSPMode
        if (user_in == "exit")
            return -1;
    }
    return -1;
}
