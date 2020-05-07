#include <iostream>
#include "AnalysisMode.h"
#include "C4Game.h"
#include "GameMode.h"
#include "TrainUtils.h"
#include "DebugMode.h"


int main()
{
    C4Game::init_variables();

    // 0 -> analysis
    // 1 -> ssp
    // 2 -> game
    // -1 -> exit
    int runcode = 0;
    for (;;)
    {
        switch (runcode)
        {
        case 0:
            runcode = AnalysisMode();
            break;
        case 1:
            runcode = SSPMode();
            break;
        case 2:
            runcode = GameMode();
            break;
        case 3:
            runcode = DBGMode();
            break;
        default:
            break;
        }
        if (runcode < 0)
            break;
    }

    return 0;
}

