#ifndef C4UCT_MODEL_H
#define C4UCT_MODEL_H

#include <onnxruntime_cxx_api.h>
#include "C4Game.h"


class Model
{
private:
    Ort::Env* env;
    Ort::SessionOptions* session_options;
    Ort::Session* session;
    Ort::AllocatorWithDefaultOptions* allocator;
    Ort::MemoryInfo* allocator_info;

    Ort::Value* input_tensor;

    std::array<const char*, 1> input_node_names;
    std::array<const char*, 2> output_node_names;

    std::array<float, 126>* input_vector;
    float* input_data;  // input_vector.data()

    float* output_values[2];

public:

    Model(Ort::Env* env, Ort::SessionOptions* session_options, Ort::Session* session,
        Ort::AllocatorWithDefaultOptions* allocator, std::array<const char*, 1> input_node_names,
        std::array<const char*, 2> output_node_names, Ort::MemoryInfo* allocator_info,
        std::array<float, 126>* input_vector, float* input_data, Ort::Value* input_tensor);

    float** Run();
    void SetPositionData(float* position_data);
    void PeekPositionData();

    ~Model();
};

#endif
