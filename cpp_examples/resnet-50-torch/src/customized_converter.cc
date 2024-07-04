/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2023/02/08
 * Brief  Customized converter of model, including pre-process and post-process.
 */

#include "customized_converter.h"

#include <torch/torch.h>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/thread/future.hpp>
#include <boost/thread/latch.hpp>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
// #include <tensorflow/core/framework/tensor.h> to include tensorflow::Tensor.
// #include <ATen/core/Tensor.h> to include at::Tensor.

#include "logger/logger.h"

namespace netease::grps {
YourConverter::YourConverter() = default;
YourConverter::~YourConverter() = default;

boost::asio::thread_pool g_decode_tp(boost::thread::hardware_concurrency());

void YourConverter::Init(const std::string& path, const YAML::Node& args) {
  Converter::Init(path, args);
  // Add your codes here.
  CLOG4(INFO, "your converter init, path: " << path << ", args: " << args);
  // Load labels.
  std::fstream input(path, std::ios::in);
  if (!input.is_open()) {
    CLOG4(ERROR, "Failed to open labels file " << path);
    throw ConverterException("Failed to open labels file " + path);
  }
  std::string line;
  while (std::getline(input, line)) {
    labels_.emplace_back(line);
  }
}
static void DecodeImgToTensor(const std::vector<char>& bytes, torch::Tensor& output) {
  cv::Mat img = cv::imdecode(bytes, cv::IMREAD_COLOR);
  if (img.empty()) {
    CLOG4(ERROR, "Failed to decode image.");
    output = torch::zeros({1, 3, 224, 224});
    return;
  }

  // Convert to RGB.
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  // Resize to 224x224.
  cv::resize(img, img, cv::Size(224, 224));
  // Convert to (0 ~ 1) float.
  img.convertTo(img, CV_32FC3, 1.0 / 255.0);
  // normalize
  cv::subtract(img, cv::Scalar(0.485, 0.456, 0.406), img);
  cv::divide(img, cv::Scalar(0.229, 0.224, 0.225), img);
  output = torch::from_blob(img.data, {img.rows, img.cols, img.channels()}, torch::kFloat32).clone();
  output = output.permute({2, 0, 1}).unsqueeze(0);
}

static void DecodeImgToTensor(const std::vector<std::vector<char>>& bytes, torch::Tensor& output) {
  std::vector<torch::Tensor> tensors(bytes.size());
  boost::latch done(bytes.size());
  for (size_t i = 0; i < bytes.size(); ++i) {
    boost::asio::post(g_decode_tp, [&bytes, &tensors, &done, i]() {
      DecodeImgToTensor(bytes[i], tensors[i]);
      done.count_down();
    });
  }
  done.wait();
  output = torch::cat(tensors, 0);
}

void YourConverter::PreProcess(const ::grps::protos::v1::GrpsMessage& input,
                               std::vector<std::pair<std::string, TensorWrapper>>& output,
                               GrpsContext& context) {
  // Extract image from bin_data.
  std::vector<char> img_data;
  for (const auto& byte : input.bin_data()) {
    img_data.emplace_back(byte);
  }
  if (img_data.empty()) {
    CLOG4(ERROR, "Input image is empty.");
    throw ConverterException("Input image is empty.");
  }
  std::shared_ptr<torch::Tensor> img_tensor = std::make_shared<at::Tensor>();
  DecodeImgToTensor(img_data, *img_tensor);
#if YOUR_CONVERTER_DEBUG
  CLOG4(INFO, "image tensor shape: " << img_tensor->shape().DebugString());
  CLOG4(INFO, "image tensor data: " << img_tensor->DebugString());
#endif
  output.emplace_back("x", img_tensor);
}

void YourConverter::PostProcess(const std::vector<std::pair<std::string, TensorWrapper>>& input,
                                ::grps::protos::v1::GrpsMessage& output,
                                GrpsContext& context) {
  // Get output tensor.
  if (input.empty()) {
    CLOG4(ERROR, "model output is empty.");
    throw ConverterException("model output is empty.");
  }
  const auto& tensor_wrapper = input[0].second;
  if (!tensor_wrapper.torch_tensor) {
    CLOG4(ERROR, "model output is empty.");
    throw ConverterException("model output is empty.");
  }
  // Get top 1 index.
  const auto& scores = tensor_wrapper.torch_tensor->to(torch::kCPU)[0];
  int max_index = 0;
  auto max_score = scores[0].item<float>();
  for (int i = 1; i < tensor_wrapper.torch_tensor->numel(); ++i) {
    auto cur_score = scores[i].item<float>();
    if (cur_score > max_score) {
      max_index = i;
      max_score = cur_score;
    }
  }
  output.set_str_data(labels_[max_index]);
}

void YourConverter::BatchPreProcess(std::vector<const ::grps::protos::v1::GrpsMessage*>& inputs,
                                    std::vector<std::pair<std::string, TensorWrapper>>& output,
                                    std::vector<GrpsContext*>& ctxs) {
  // CLOG4(INFO, "BatchPreProcess, batch size: " << inputs.size());
  std::vector<std::vector<char>> img_datas;
  for (const auto& input : inputs) {
    std::vector<char> img_data;
    for (const auto& byte : input->bin_data()) {
      img_data.emplace_back(byte);
    }
    if (img_data.empty()) {
      CLOG4(ERROR, "Input image is empty.");
      throw ConverterException("Input image is empty.");
    }
    img_datas.emplace_back(img_data);
  }
  std::shared_ptr<torch::Tensor> img_tensor = std::make_shared<at::Tensor>();
  DecodeImgToTensor(img_datas, *img_tensor);

#if YOUR_CONVERTER_DEBUG
  CLOG4(INFO, "image tensor shape: " << img_tensor->shape().DebugString());
  CLOG4(INFO, "image tensor data: " << img_tensor->DebugString());
#endif
  output.emplace_back("x", img_tensor);
}

void YourConverter::BatchPostProcess(const std::vector<std::pair<std::string, TensorWrapper>>& input,
                                     std::vector<::grps::protos::v1::GrpsMessage*>& outputs,
                                     std::vector<GrpsContext*>& ctxs) {
  // CLOG4(INFO, "BatchPostProcess, batch size: " << outputs.size());
  // Get output tensor.
  if (input.empty()) {
    CLOG4(ERROR, "model output is empty.");
    throw ConverterException("model output is empty.");
  }
  const auto& tensor_wrapper = input[0].second;
  if (!tensor_wrapper.torch_tensor) {
    CLOG4(ERROR, "model output is empty.");
    throw ConverterException("model output is empty.");
  }
  // Get top 1 index.
  const auto& scores = tensor_wrapper.torch_tensor->to(torch::kCPU);
  for (int i = 0; i < scores.size(0); ++i) {
    int max_index = 0;
    auto max_score = scores[i][0].item<float>();
    for (int j = 1; j < scores.size(1); ++j) {
      auto cur_score = scores[i][j].item<float>();
      if (cur_score > max_score) {
        max_index = j;
        max_score = cur_score;
      }
    }
    outputs[i]->set_str_data(labels_[max_index]);
  }
}

// Implement and register other converter class after here.

} // namespace netease::grps
