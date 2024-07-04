/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2023/02/08
 * Brief  Customized converter of model, including pre-process and post-process.
 */

#include "customized_converter.h"

#include <tensorflow/core/framework/tensor.h>

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

boost::asio::thread_pool g_decode_tp(boost::thread::hardware_concurrency());

namespace netease::grps {
YourConverter::YourConverter() = default;

YourConverter::~YourConverter() = default;

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

static void DecodeImgToTensor(const std::vector<char>& bytes, tensorflow::Tensor& output) {
  cv::Mat img = cv::imdecode(bytes, cv::IMREAD_COLOR);
  if (img.empty()) {
    CLOG4(ERROR, "Failed to decode image.");
    throw Converter::ConverterException("Failed to decode image.");
  }

  // Convert to RGB.
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  // Resize to 224x224.
  cv::resize(img, img, cv::Size(224, 224));
  // Convert to (0 ~ 1) float.
  img.convertTo(img, CV_32FC3, 1.0 / 255.0);

  // Convert to tensor.
  output = tensorflow::Tensor(tensorflow::DT_FLOAT, {1, 224, 224, 3});
  auto tensor_mapped = output.tensor<float, 4>();
  for (int y = 0; y < 224; ++y) {
    for (int x = 0; x < 224; ++x) {
      auto pixel = img.at<cv::Vec3f>(y, x);
      tensor_mapped(0, y, x, 0) = pixel[0];
      tensor_mapped(0, y, x, 1) = pixel[1];
      tensor_mapped(0, y, x, 2) = pixel[2];
    }
  }
}

static void DecodeImgTTensor(const std::vector<std::vector<char>>& bytes, tensorflow::Tensor& output) {
  output = tensorflow::Tensor(tensorflow::DT_FLOAT, {int64_t(bytes.size()), 224, 224, 3});

  boost::latch done(bytes.size());
  for (int i = 0; i < bytes.size(); ++i) {
    boost::asio::post(g_decode_tp, [&bytes, &done, &output, i]() {
      cv::Mat img = cv::imdecode(bytes[i], cv::IMREAD_COLOR);
      if (img.empty()) {
        CLOG4(ERROR, "Failed to decode image.");
        done.count_down();
        return;
      }

      // Convert to RGB.
      cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
      // Resize to 224x224.
      cv::resize(img, img, cv::Size(224, 224));
      // Convert to (0 ~ 1) float.
      img.convertTo(img, CV_32FC3, 1.0 / 255.0);

      auto tensor_mapped = output.tensor<float, 4>();
      for (int y = 0; y < 224; ++y) {
        for (int x = 0; x < 224; ++x) {
          auto pixel = img.at<cv::Vec3f>(y, x);
          tensor_mapped(i, y, x, 0) = pixel[0];
          tensor_mapped(i, y, x, 1) = pixel[1];
          tensor_mapped(i, y, x, 2) = pixel[2];
        }
      }
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
    CLOG4(ERROR, "image data is empty.");
    throw ConverterException("image data is empty.");
  }

  std::shared_ptr<tensorflow::Tensor> img_tensor = std::make_shared<tensorflow::Tensor>();
  DecodeImgToTensor(img_data, *img_tensor);

#if YOUR_CONVERTER_DEBUG
  CLOG4(INFO, "image tensor shape: " << img_tensor->shape().DebugString());
  CLOG4(INFO, "image tensor data: " << img_tensor->DebugString());
#endif

  output.emplace_back("inputs", img_tensor);
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
  if (!tensor_wrapper.tf_tensor) {
    CLOG4(ERROR, "model output tensor is empty.");
    throw ConverterException("model output tensor is empty.");
  }

  // Find the top 1 index.
  const auto& scores = tensor_wrapper.tf_tensor->flat<float>().data();
  int max_index = 0;
  float max_score = scores[0];
  for (int i = 1; i < tensor_wrapper.tf_tensor->NumElements(); ++i) {
    if (scores[i] > max_score) {
      max_index = i;
      max_score = scores[i];
    }
  }

  // Set output.
  output.set_str_data(labels_[max_index]);
}

void YourConverter::BatchPreProcess(std::vector<const ::grps::protos::v1::GrpsMessage*>& inputs,
                                    std::vector<std::pair<std::string, TensorWrapper>>& output,
                                    std::vector<GrpsContext*>& ctxs) {
  std::vector<std::vector<char>> img_datas;
  for (const auto& input : inputs) {
    std::vector<char> img_data;
    for (const auto& byte : input->bin_data()) {
      img_data.emplace_back(byte);
    }
    if (img_data.empty()) {
      CLOG4(ERROR, "image data is empty.");
      throw ConverterException("image data is empty.");
    }
    img_datas.emplace_back(img_data);
  }

  std::shared_ptr<tensorflow::Tensor> img_tensor = std::make_shared<tensorflow::Tensor>();
  DecodeImgTTensor(img_datas, *img_tensor);

#if YOUR_CONVERTER_DEBUG
  CLOG4(INFO, "image tensor shape: " << img_tensor->shape().DebugString());
  CLOG4(INFO, "image tensor data: " << img_tensor->DebugString());
#endif

  output.emplace_back("inputs", img_tensor);
}

void YourConverter::BatchPostProcess(const std::vector<std::pair<std::string, TensorWrapper>>& input,
                                     std::vector<::grps::protos::v1::GrpsMessage*>& outputs,
                                     std::vector<GrpsContext*>& ctxs) {
  // Get output tensor.
  if (input.empty()) {
    CLOG4(ERROR, "model output is empty.");
    throw ConverterException("model output is empty.");
  }
  const auto& tensor_wrapper = input[0].second;
  if (!tensor_wrapper.tf_tensor) {
    CLOG4(ERROR, "model output tensor is empty.");
    throw ConverterException("model output tensor is empty.");
  }

  // Find the top 1 in batch.
  std::vector<int> max_index;
  int64_t batch_size = tensor_wrapper.tf_tensor->dim_size(0);
  for (int64_t i = 0; i < batch_size; ++i) {
    const auto& scores = tensor_wrapper.tf_tensor->tensor<float, 2>();
    int max_idx = 0;
    float max_score = scores(i, 0);
    for (int j = 1; j < tensor_wrapper.tf_tensor->dim_size(1); ++j) {
      if (scores(i, j) > max_score) {
        max_idx = j;
        max_score = scores(i, j);
      }
    }
    max_index.emplace_back(max_idx);
  }

  // Set output.
  for (int i = 0; i < batch_size; ++i) {
    outputs[i]->set_str_data(labels_[max_index[i]]);
  }
}

// Implement and register other converter class after here.

} // namespace netease::grps
