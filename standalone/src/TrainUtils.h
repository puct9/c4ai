#ifndef C4UCT_TRAIN_UTILS_H
#define C4UCT_TRAIN_UTILS_H

#include "gsl/gsl_rng.h"
#include "Model.h"


int SSPMode();
void StochasticSelfPlay(Model* network, float c_puct, float dir_alpha, int temp_cutoff, size_t playouts, gsl_rng* rng);


#endif
