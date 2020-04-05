#include "ModelManager.h"
#include "Model.h"


ModelManager::ModelManager()
{
}

Model * ModelManager::CreateModel(const wchar_t * model_path)
{
    this->envs.push_back(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "mdl"));
    Ort::Env* env = &this->envs[this->envs.size() - 1];

    this->session_options.push_back(Ort::SessionOptions());
    Ort::SessionOptions* session_options = &this->session_options[this->session_options.size() - 1];
    session_options->SetThreadPoolSize(1);
    session_options->SetGraphOptimizationLevel(2);

    this->sessions.push_back(Ort::Session(*env, model_path, *session_options));
    Ort::Session* session = &this->sessions[this->sessions.size() - 1];

    this->allocators.push_back(Ort::Allocator::CreateDefault());
    Ort::Allocator* allocator = &this->allocators[this->allocators.size() - 1];

    // get input information
    std::array<const char*, 1> input_node_names = { session->GetInputName(0, *allocator) };
    this->input_node_names.push_back(input_node_names);

    // get output information
    std::array<const char*, 2> output_node_names;
    for (int i = 0; i < 2; i++)
    {
        output_node_names[i] = session->GetOutputName(i, *allocator);
    }
    this->output_node_names.push_back(output_node_names);

    // allocator
    this->allocator_infos.push_back(Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    Ort::AllocatorInfo* allocator_info = &this->allocator_infos[this->allocator_infos.size() - 1];

    // manage INPUT vector data and tensors
    this->input_vectors.push_back(std::array<float, 126>());
    float* input_data = this->input_vectors[this->input_vectors.size() - 1].data();

    std::array<int64_t, 4> input_node_dims = { 1, 7, 6, 3 };
    this->input_node_dims.push_back(input_node_dims);
    this->input_tensors.push_back(Ort::Value::CreateTensor<float>(*allocator_info, input_data, 126,
        this->input_node_dims[this->input_node_dims.size() - 1].data(), 4));
    Ort::Value* input_tensor = &this->input_tensors[this->input_tensors.size() - 1];

    this->models.push_back(Model(env, session_options, session, allocator, 
        this->input_node_names[this->input_node_names.size() - 1],
        this->output_node_names[this->output_node_names.size() - 1],
        allocator_info, &this->input_vectors[this->input_vectors.size() - 1],
        input_data, input_tensor));

    return &this->models[this->models.size() - 1];
}


ModelManager::~ModelManager()
{
}
