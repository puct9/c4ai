#include "stdafx.h"
#include <iostream>
#include <string>
#include <sstream>
#include "GameMode.h"
#include "MCTSNode.h"
#include "MCTSEngine.h"


void Gamemode()
{
    ModelManager model_manager;

    std::string user_in;
    size_t playouts;
    std::cout << "Search playouts (default 5000): ";
    std::getline(std::cin, user_in);
    user_in = user_in.size() == 0 ? "5000" : user_in;
    std::stringstream(user_in) >> playouts;

    playouts = playouts == 0 ? 5000 : playouts;
    playouts = playouts < 10 ? 10 : playouts;
    std::cout << "Set playouts to " << playouts << std::endl;
    std::cout << "We will need about " << playouts * 8 * sizeof(MCTSNode) / (float)(1024 * 1024)
        << " MB of RAM." << std::endl;

    std::cout << "Model (default save_2071.onnx): ";
    std::getline(std::cin, user_in);
    user_in = user_in.size() == 0 ? "save_2071.onnx" : user_in;
    std::string str_mdl_name = "Models/" + user_in;
    std::cout << "Using model: " << str_mdl_name << std::endl;
    std::wstring wstr_mdl_name = std::wstring(str_mdl_name.begin(), str_mdl_name.end());
    const wchar_t* mdl_name = wstr_mdl_name.c_str();
    std::cout.flush();

    Model* MODEL = model_manager.CreateModel(mdl_name);

    C4Game game;


    while (game.GameOver() == -1)
    {
        game.Show();
        int move_n = game.GetMoveNum();
        if (move_n % 2 == 0)
        {
            human_controller(game, MODEL, playouts);
        }
        else
        {
            computer_controller(game, MODEL, playouts);
        }
    }

    std::cout << "Game over!" << std::endl;
    game.Show();
}


void human_controller(C4Game& state, Model* mdl, size_t playouts)
{
    int move_inp;
    bool* legal = state.LegalMoves();
    for (;;)
    {
        std::cout << "Your turn: ";
        std::string inp;
        std::getline(std::cin, inp);
        std::cout.flush();
        if (inp == "go")
        {
            computer_controller(state, mdl, playouts);
            return;
        }
        std::stringstream(inp) >> move_inp;
        if (0 <= move_inp && move_inp <= 6)
            if (legal[move_inp])
                break;
    }
    state.PlayMove(move_inp);
}


void computer_controller(C4Game& state, Model* mdl, size_t playouts)
{
    // 30000N
    MCTSEngine eng = MCTSEngine(C4Game(state), mdl, 3, playouts);
    eng.DoPlayouts();
    std::vector<int> pv = eng.GetPV();
    std::cout << "Winrate: " << eng.GetTopNode()->GetChild(pv[0])->GetQ() * 50 + 50 <<
        " %" << std::endl;
    state.PlayMove(pv[0]);
}
