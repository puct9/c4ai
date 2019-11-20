#pragma once
#include "C4Game.h"
#include "Model.h"
#include "ModelManager.h"


void Gamemode();

void human_controller(C4Game& state, Model* mdl, size_t playouts);
void computer_controller(C4Game& state, Model* mdl, size_t playouts);
