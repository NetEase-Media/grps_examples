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
  ModelInferer::Init(path, device, args);
  // Add your codes here.
  CLOG4(INFO, "your inferer init success. path: " << path << ", device: " << device << ", args: " << args);
}

void YourInferer::Load() {
  // Add your codes here.
  CLOG4(INFO, "your inferer load success.");
}

void YourInferer::Infer(const std::vector<std::pair<std::string, TensorWrapper>>& inputs,
                        std::vector<std::pair<std::string, TensorWrapper>>& outputs,
                        GrpsContext& context) {
  auto* cntl = context.http_controller();

  if (inputs.size() != 2) {
    // context.set_has_err(true);
    context.set_err_msg("inputs size should be 2.");
    cntl->http_response().set_status_code(brpc::HTTP_STATUS_BAD_REQUEST);
    cntl->http_response().set_content_type("text/plain");
    cntl->response_attachment().append(context.err_msg());
    LOG4(ERROR, "YourInferer::Infer failed, err msg: " << context.err_msg());
    return;
  }

  auto a = inputs[0].second.float_tensor;
  auto b = inputs[1].second.float_tensor;
  auto c = a + b;

  outputs.emplace_back("c", TensorWrapper(c));
}
} // namespace netease::grps
