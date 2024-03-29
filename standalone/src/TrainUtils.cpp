#include <algorithm>
#include <cfloat>
#include <math.h>
#include <iostream>
#include <sstream>
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "TrainUtils.h"
#include "C4Game.h"
#include "MCTSEngine.h"
#include "ModelManager.h"


int SSPMode()
{
    ModelManager model_manager;
    std::cout.flush();

    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);

    // default selfplay parameters that may be manually changed
    float c_puct = 3.0f;
    float dir_alpha = 1.3f;
    int temp_cutoff = 12;
    std::uint64_t playouts = 800;

#ifdef _WIN32
    Model* model = model_manager.CreateModel(L"Models/temp.onnx");
#else
    Model* model = model_manager.CreateModel("Models/temp.onnx");
#endif
    C4Game game;

    std::cout << "Welcome to selfplay mode." << std::endl;

    // input loop
    std::string user_in;
    unsigned long user_value;
    for (;;)
    {
        std::getline(std::cin, user_in);
        if (user_in.size() == 0)
            continue;
        if (user_in == "game")
            return 2;
        if (user_in == "isready")
            std::cout << "readyok" << std::endl;
        if (user_in.rfind("seed ", 0) == 0)
        {
            std::string number = user_in.substr(5);
            std::stringstream(number) >> user_value;
            gsl_rng_set(rng, user_value);
            std::cout << "seed set to " << user_value << std::endl;
        }
        // parameter configuration options
        if (user_in.rfind("c_puct set ", 0) == 0)
        {
            std::string number = user_in.substr(11);
            c_puct = std::stof(number);
        }
        if (user_in.rfind("dir_alpha set ", 0) == 0)
        {
            std::string number = user_in.substr(14);
            dir_alpha = std::stof(number);
        }
        if (user_in.rfind("temp_cutoff set ", 0) == 0)
        {
            std::string number = user_in.substr(16);
            std::stringstream(number) >> temp_cutoff;
        }
        if (user_in.rfind("playouts set ", 0) == 0)
        {
            std::string number = user_in.substr(13);
            std::stringstream(number) >> playouts;
        }
        if (user_in == "params")
        {
            std::cout << "Parameters\nc_puct " << c_puct << "\ndir_alpha " << dir_alpha << "\ntemp_cutoff " << temp_cutoff << "\nplayouts " << playouts << std::endl;
        }
        if (user_in == "sspgo")
            StochasticSelfPlay(model, c_puct, dir_alpha, temp_cutoff, playouts, rng);
        if (user_in == "exit")
            return -1;
    }
    return -1;
}

void StochasticSelfPlay(Model* network, float c_puct, float dir_alpha, int temp_cutoff, std::uint64_t playouts, gsl_rng* rng)
{
    // prepare dirichlet
    double alpha[7];
    double theta[7];  // where the return values of the dirichlet function go
    std::fill_n(alpha, 7, dir_alpha);

    C4Game board;
    MCTSEngine eng(board, network, c_puct, playouts);
    int move_n = 0;
    while (board.GameOver() == -1)
    {
        eng.DoPlayouts();
        float* probs = eng.GetMoveProbs();

        // get some dirichlet noise with alpha
        bool* legals = board.LegalMoves();
        int required_dirichlet_n = 0;
        for (int i = 0; i < 7; i++)
        {
            if (legals[i])
                required_dirichlet_n++;
        }
        gsl_ran_dirichlet(rng, required_dirichlet_n, alpha, theta);

        // write the probs
        int curr_move = 0;
        double probs_copy[7] = { 0 };
        for (int i = 0; i < 7; i++)
        {
            if (legals[i])
            {
                probs_copy[i] = 0.84L * probs[i] + 0.16L * theta[curr_move];
                curr_move++;
            }
        }

        // apply temperature
        // apply log and divide by temperature and apply exponent
        double temperature = (move_n < temp_cutoff) ? 1L : 0.05L;
        double probs_sum = 0;
        for (int i = 0; i < 7; i++)
        {
            if (legals[i])
            {
                probs_copy[i] = exp(log(probs_copy[i] + DBL_MIN) / temperature);
                probs_sum += probs_copy[i];
            }
        }
        // normalise the array
        for (int i = 0; i < 7; i++)
        {
            probs_copy[i] = probs_copy[i] / probs_sum;
        }

        // logging
        if (true)
        {
            for (int i = 0; i < 7; i++)
            {
                std::cout << probs[i] << ' ';
            }
        }

        // select a move
        double choice_random = gsl_rng_uniform(rng);
        double accumulator = 0;
        for (int i = 0; i < 7; i++)
        {
            accumulator += probs_copy[i];
            if (accumulator > choice_random)
            {
                board.PlayMove(i);
                eng.RecycleTree(i);
                move_n++;
                std::cout << '~' << i << std::endl;
                break;
            }
        }
    }
    std::cout << "done" << std::endl;
}
