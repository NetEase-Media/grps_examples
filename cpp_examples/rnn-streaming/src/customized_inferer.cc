/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2023/02/15
 * Brief  Customized deep learning model inferer. Including model load and model infer.
 */

#include "customized_inferer.h"

#include "logger/logger.h"

namespace netease::grps {
YourInferer::YourInferer() = default;
YourInferer::~YourInferer() = default;

void YourInferer::Init(const std::string& path, const std::string& device, const YAML::Node& args) {
  TorchModelInferer::Init(path, device, args);
  pred_len_ = args["pred_length"].as<int>();
  CLOG4(INFO, "your inferer init success. path: " << path << ", device: " << device << ", args: " << args);
}

void YourInferer::Load() {
  TorchModelInferer::Load();
  std::ifstream file("./data/word_to_ix.txt");
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string word, idx;
    iss >> word >> idx;
    word_to_idx_[word] = std::stoi(idx);
    idx_to_word_[std::stoi(idx)] = word;
  }
  file.close();
  CLOG4(INFO, "your inferer load success.");
}

void YourInferer::Infer(const ::grps::protos::v1::GrpsMessage& inputs,
                        ::grps::protos::v1::GrpsMessage& outputs,
                        netease::grps::GrpsContext& ctx) {
  // Init first input.
  std::string inputs_s = inputs.str_data();
  std::vector<long> prime_input;
  std::stringstream ss(inputs_s);
  std::string word;
  while (ss >> word) { // Skip space.
    prime_input.push_back(word_to_idx_[word]);
  }
  if (prime_input.size() != 2) { // prime_input size must be 2.
    throw InfererException("Input text must be 2 words.");
  }

  // Return first output.
  if (ctx.IfStreaming()) {
    outputs.set_str_data(inputs_s);
    ctx.StreamingRespond(outputs);
  }

  std::vector<std::pair<std::string, TensorWrapper>> tensor_wrapper_input;
  std::vector<std::pair<std::string, TensorWrapper>> tensor_wrapper_output;

  // Build input tensor.
  std::shared_ptr<torch::Tensor> p_inp = std::make_shared<torch::Tensor>(
    torch::from_blob(prime_input.data(), {static_cast<long>(prime_input.size())}, torch::kLong));
  std::shared_ptr<torch::Tensor> p_hidden = std::make_shared<torch::Tensor>(torch::zeros({1, 1, 100}, torch::kFloat32));
  tensor_wrapper_input.emplace_back("input", p_inp);
  tensor_wrapper_input.emplace_back("hidden", p_hidden);

  for (int i = 0; i < pred_len_; i++) {
    // infer
    TorchModelInferer::Infer(tensor_wrapper_input, tensor_wrapper_output, ctx);

    // postprocess
    // Sample from the network as a multinomial distribution
    torch::Tensor output_dist = tensor_wrapper_output[0].second.torch_tensor->data().view({-1}).div(1).exp();
    long top_i = torch::multinomial(output_dist, 1).item<long>();

    // Add predicted character to string and use as next input
    if (ctx.IfStreaming()) {
      if (ctx.IfDisconnected()) {
        break;
      }
      outputs.set_str_data(idx_to_word_[top_i]);
      ctx.StreamingRespond(outputs);
    } else {
      inputs_s += " " + idx_to_word_[top_i];
    }

    // Update next input tensor.
    prime_input[0] = prime_input[1];
    prime_input[1] = top_i;
    *p_hidden = std::move(*tensor_wrapper_output[1].second.torch_tensor);
  }

  if (!ctx.IfStreaming()) {
    outputs.set_str_data(inputs_s);
  }
}
} // namespace netease::grps
