/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2023/02/15
 * Brief  Customized deep learning model inferer. Including model load and model infer.
 */

#pragma once

#include "model_infer/inferer.h"
#include "model_infer/torch_inferer.h"

namespace netease::grps {
class YourInferer : public TorchModelInferer {
public:
  YourInferer();
  ~YourInferer() override;

  // Clone inferer for duplicated use. Don't edit this function.
  ModelInferer* Clone() override { return new YourInferer(); }

  /**
   * @brief Init model inferer.
   * @param path: Model path, it can be a file path or a directory path.
   * @param device: Device to run model.
   * @param args: More args.
   * @throw InfererException: If init failed, can throw InfererException and will be caught by server and show error
   * message to user when start service.
   */
  void Init(const std::string& path, const std::string& device, const YAML::Node& args) override;

  /**
   * @brief Load model.
   * @throw InfererException: If load failed, can throw InfererException and will be caught by server and show error
   * message to user when start service.
   */
  void Load() override;

  /**
   * Used when in `no converter mode`. Input and output are directly GrpsMessage.
   * @brief Infer model.
   * @param input: Input.
   * @param output: Output.
   * @param ctx: Context of current request.
   * @throw InfererException: If infer failed, can throw InfererException and will be caught by server and return error
   * message to client.
   */
  void Infer(const ::grps::protos::v1::GrpsMessage& input,
             ::grps::protos::v1::GrpsMessage& output,
             GrpsContext& ctx) override;

private:
  int pred_len_;
  std::map<std::string, long> word_to_idx_;
  std::map<long, std::string> idx_to_word_;
};

REGISTER_INFERER(YourInferer, your_inferer); // Register your inferer.

// Define other inferer class after here.

} // namespace netease::grps