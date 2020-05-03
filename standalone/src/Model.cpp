#include <onnxruntime_cxx_api.h>
#include "Model.h"
#include <iostream>



Model::Model(Ort::Env* env, Ort::SessionOptions* session_options, Ort::Session* session,
    Ort::AllocatorWithDefaultOptions* allocator, std::array<const char*, 1> input_node_names,
    std::array<const char*, 2> output_node_names, Ort::MemoryInfo* allocator_info,
    std::array<float, 126>* input_vector, float* input_data, Ort::Value* input_tensor)
{
    this->env = env;
    this->session_options = session_options;
    this->session = session;
    this->allocator = allocator;
    this->input_node_names = input_node_names;
    this->output_node_names = output_node_names;
    this->allocator_info = allocator_info;

    this->input_vector = input_vector;
    this->input_data = input_data;
    
    this->input_tensor = input_tensor;
}


float ** Model::Run()
{
    auto outputs = this->session->Run(Ort::RunOptions{ nullptr }, this->input_node_names.data(), this->input_tensor,
        1, this->output_node_names.data(), 2);
    this->output_values[0] = outputs[0].GetTensorMutableData<float>(); // value
    this->output_values[1] = outputs[1].GetTensorMutableData<float>(); // policy
    return this->output_values;
}

void Model::SetPositionData(float * position_data)
{
    for (int i = 0; i < 126; i++)
    {
        this->input_data[i] = position_data[i];
    }
}

void Model::PeekPositionData()
{
    std::cout << this->input_data[0] << std::endl;
}


Model::~Model()
{
}
