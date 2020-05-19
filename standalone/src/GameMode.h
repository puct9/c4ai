#ifndef C4UCT_GAME_MODE_H
#define C4UCT_GAME_MODE_H

#include <cstddef>
#include "C4Game.h"
#include "Model.h"
#include "ModelManager.h"


int GameMode();

void human_controller(C4Game& state, Model* mdl, std::uint64_t playouts);
void computer_controller(C4Game& state, Model* mdl, std::uint64_t playouts);

#endif
