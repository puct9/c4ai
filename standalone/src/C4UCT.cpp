// C4UCT.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "C4Game.h"
#include "MCTSNode.h"
#include "Model.h"
#include "ModelManager.h"
#include "MCTSEngine.h"
#include "NodeHashtable.h"
#include "GameMode.h"
#include "AnalysisMode.h"

#include <sstream>
#include <vector>
#include <string>



int main()
{
    C4Game::init_variables();
    
    for (;;)
    {
        Analysismode();
        Gamemode();
        break; // uhh
    }

    return 0;
}

