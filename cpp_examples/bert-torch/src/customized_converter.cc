/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2023/02/08
 * Brief  Customized converter of model, including pre-process and post-process.
 */

#include "customized_converter.h"

#include <torch/torch.h>

#include "logger/logger.h"

namespace netease::grps {
YourConverter::YourConverter() = default;

YourConverter::~YourConverter() = default;

void YourConverter::Init(const std::string& path, const YAML::Node& args) {
  Converter::Init(path, args);
  auto_tokenizer_ = std::make_shared<tokenizer::Tokenizer>(path_, true, true);
  CLOG4(INFO, "your converter init, path: " << path << ", args: " << args);

  // Parse more args from yaml node.
  max_length_ = args["max_length"].as<int>();
  mask_token_id_ = args["mask_token_id"].as<int>();
}

void YourConverter::PreProcess(const ::grps::protos::v1::GrpsMessage& input,
                               std::vector<std::pair<std::string, TensorWrapper>>& output,
                               GrpsContext& context) {
  const std::string& text = input.str_data();
  std::vector<std::string> tokens;
  std::vector<tokenizer::SizeT> offsets;
  auto_tokenizer_->wordpiece_tokenize(text, tokens, offsets);

  std::vector<tokenizer::SizeT> input_ids;
  std::vector<tokenizer::SizeT> token_type_ids;
  std::vector<tokenizer::SizeT> attention_mask;
  bool add_cls_sep = true;
  bool truncation = true;
  tokenizer::SizeT max_length = max_length_;
  auto_tokenizer_->encode(text, input_ids, token_type_ids, attention_mask, offsets, add_cls_sep, truncation,
                          max_length);
  std::vector<int> mask_pos;
  std::vector<long> input_id_long;
  for (int i = 0; i < input_ids.size(); ++i) {
    input_id_long.push_back((long)(input_ids[i]));
    // mask position.
    if (input_ids[i] == mask_token_id_) {
      mask_pos.push_back(i);
    }
  }

  torch::Tensor outTensor =
    torch::from_blob(input_id_long.data(), {1, (int)input_id_long.size()}, torch::kInt64).clone();

  output.emplace_back("input_ids", std::move(outTensor));

  // Put mask position to context.
  CLOG4(INFO, "your converter pre process. mask_pos: " << mask_pos);
  context.SetUserData(std::move(mask_pos));
}

void YourConverter::PostProcess(const std::vector<std::pair<std::string, TensorWrapper>>& input,
                                ::grps::protos::v1::GrpsMessage& output,
                                GrpsContext& context) {
  if (input.empty()) {
    CLOG4(INFO, "PostProcess input empty, return ");
    return;
  }

  // Get mask position from context.
  auto& mask_pos = context.GetUserData<std::vector<int>>();
  CLOG4(INFO, "your converter post process. mask_pos: " << mask_pos);
  if (mask_pos.empty()) {
    CLOG4(INFO, "mask is empty, return ");
    return;
  }

  auto tensor = *input[0].second.torch_tensor; // {1 * 21 * 21128}
  auto tensor0 = tensor.select(0, 0);
  auto max = torch::argmax(tensor0, 1);

  std::vector<tokenizer::SizeT> tokenIds;
  for (const auto& item : mask_pos) {
    int tokenId = max[item].item<int>();
    tokenIds.push_back(tokenId);
  }
  auto result = auto_tokenizer_->convert_ids_to_tokens(tokenIds);

  std::string str_result;
  for (const auto& s : result) {
    if (!str_result.empty()) {
      str_result += "||";
    }
    str_result += s;
  }
  output.set_str_data(str_result);
}

// Implement and register other converter class after here.

} // namespace netease::grps
