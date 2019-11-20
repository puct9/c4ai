#pragma once
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include "Model.h"


class ModelManager
{
private:
    std::vector<Model> models;

    // we need to keep track of a bunch of stuff to support the models
    // all of this is done to make a very nice high level API and prevent
    // dangling references and pointers
    
    std::vector <Ort::Env> envs;
    std::vector<Ort::SessionOptions> session_options;
    std::vector<Ort::Session> sessions;
    std::vector<Ort::Allocator> allocators;
    std::vector<Ort::AllocatorInfo> allocator_infos;

    std::vector<Ort::Value> input_tensors;

    std::vector<std::array<const char*, 1>> input_node_names;
    std::vector<std::array<const char*, 2>> output_node_names;

    std::vector<float*> input_datas;
    
    std::vector<std::array<float, 126>> input_vectors;
    std::vector<std::array<int64_t, 4>> input_node_dims;

public:
    ModelManager();

    Model* CreateModel(const wchar_t * model_path);

    ~ModelManager();
};

