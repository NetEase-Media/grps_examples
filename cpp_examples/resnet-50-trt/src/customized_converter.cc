/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2024/06/28
 * Brief  Customized converter of model, including pre-process and post-process.
 */

#include "customized_converter.h"

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/thread/future.hpp>
#include <boost/thread/latch.hpp>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "logger/logger.h"

#define YOUR_CONVERTER_DEBUG 0

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
    // Get label between ''
    auto label = line.substr(line.find_first_of('\'') + 1, line.find_last_of('\'') - line.find_first_of('\'') - 1);
    labels_.emplace_back(label);
  }
}

static void DecodeImgToTensor(const std::vector<char>& bytes, TrtHostBinding& tensor, size_t idx) {
  auto* buffer = (float*)tensor.buffer().Get() + idx * 3 * 224 * 224;

  cv::Mat img = cv::imdecode(bytes, cv::IMREAD_COLOR);
  if (img.empty()) {
    CLOG4(ERROR, "Failed to decode image.");
    memset(buffer, 0, 3 * 224 * 224 * sizeof(float));
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
  // Permute to CHW.
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < 224; ++h) {
      for (int w = 0; w < 224; ++w) {
        buffer[c * 224 * 224 + h * 224 + w] = img.at<cv::Vec3f>(h, w)[c];
      }
    }
  }
}

static void DecodeImgToTensor(const std::vector<std::vector<char>>& bytes, TrtHostBinding& tensor) {
  boost::latch done(bytes.size());
  for (size_t i = 0; i < bytes.size(); ++i) {
    boost::asio::post(g_decode_tp, [&bytes, &tensor, &done, i]() {
      DecodeImgToTensor(bytes[i], tensor, i);
      done.count_down();
    });
  }
  done.wait();
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
  auto img_tensor =
    std::make_shared<TrtHostBinding>("x", nvinfer1::Dims({4, {1, 3, 224, 224}, {}}), nvinfer1::DataType::kFLOAT);
  DecodeImgToTensor(img_data, *img_tensor, 0);
#if YOUR_CONVERTER_DEBUG
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
  if (!tensor_wrapper.trt_host_binding) {
    CLOG4(ERROR, "model output is empty.");
    throw ConverterException("model output is empty.");
  }
  // Get top 1 index.
  const auto* scores = (float*)tensor_wrapper.trt_host_binding->buffer().Get();
  int max_index = 0;
  auto max_score = scores[0];
  for (int i = 1; i < tensor_wrapper.trt_host_binding->volume(); ++i) {
    auto cur_score = scores[i];
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

  auto img_tensor = std::make_shared<TrtHostBinding>("x",
                                                     nvinfer1::Dims({4,
                                                                     {
                                                                       int(inputs.size()),
                                                                       3,
                                                                       224,
                                                                       224,
                                                                     },
                                                                     {}}),
                                                     nvinfer1::DataType::kFLOAT);
  DecodeImgToTensor(img_datas, *img_tensor);

#if YOUR_CONVERTER_DEBUG
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
  if (!tensor_wrapper.trt_host_binding) {
    CLOG4(ERROR, "model output is empty.");
    throw ConverterException("model output is empty.");
  }
  // Get top 1 index.
  auto& tensor = tensor_wrapper.trt_host_binding;
  const auto* scores = (float*)tensor->buffer().Get();
  for (int i = 0; i < tensor->dims().d[0]; ++i) {
    int max_index = 0;
    auto max_score = scores[i * tensor->dims().d[1]];
    for (int j = 1; j < tensor->dims().d[1]; ++j) {
      auto cur_score = scores[i * tensor->dims().d[1] + j];
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
