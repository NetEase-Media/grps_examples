/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2023/02/08
 * Brief  Customized converter of model, including pre-process and post-process.
 */

#include "customized_converter.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "logger/logger.h"

namespace netease::grps {

YourConverter::YourConverter() = default;
YourConverter::~YourConverter() = default;

void YourConverter::Init(const std::string& path, const YAML::Node& args) {
  Converter::Init(path, args);
  // Add your codes here.
  CLOG4(INFO, "your converter init, path: " << path << ", args: " << args);
}

void YourConverter::PreProcess(const ::grps::protos::v1::GrpsMessage& input,
                               std::vector<std::pair<std::string, TensorWrapper>>& output,
                               GrpsContext& context) {
  auto* cntl = context.http_controller();

  // Check content type.
  const auto& content_type = cntl->http_request().content_type();
  if (content_type != "application/json") {
    // context.set_has_err(true);
    context.set_err_msg("Unsupported media type: " + content_type);
    cntl->http_response().set_status_code(brpc::HTTP_STATUS_BAD_REQUEST);
    cntl->http_response().set_content_type("text/plain");
    cntl->response_attachment().append(context.err_msg());
    CLOG4(ERROR, "YourConverter::PreProcess failed, err msg: " << context.err_msg());
    return;
  }

  // Parse "a" and "b" from json.
  rapidjson::Document doc;
  doc.Parse(cntl->request_attachment().to_string().c_str());
  if (doc.HasParseError()) {
    // context.set_has_err(true);
    context.set_err_msg("Parse json failed.");
    cntl->http_response().set_status_code(brpc::HTTP_STATUS_BAD_REQUEST);
    cntl->http_response().set_content_type("text/plain");
    cntl->response_attachment().append(context.err_msg());
    CLOG4(ERROR, "YourConverter::PreProcess failed, err msg: " << context.err_msg());
    return;
  }

  if (!doc.HasMember("a") || !doc.HasMember("b")) {
    // context.set_has_err(true);
    context.set_err_msg("Json format error.");
    cntl->http_response().set_status_code(brpc::HTTP_STATUS_BAD_REQUEST);
    cntl->http_response().set_content_type("text/plain");
    cntl->response_attachment().append(context.err_msg());
    CLOG4(ERROR, "YourConverter::PreProcess failed, err msg: " << context.err_msg());
    return;
  }
  auto a = doc["a"].GetFloat();
  auto b = doc["b"].GetFloat();

  TensorWrapper a_tensor(a);
  TensorWrapper b_tensor(b);
  output.emplace_back("a", std::move(a_tensor));
  output.emplace_back("b", std::move(b_tensor));
}

void YourConverter::PostProcess(const std::vector<std::pair<std::string, TensorWrapper>>& input,
                                ::grps::protos::v1::GrpsMessage& output,
                                GrpsContext& context) {
  auto* cntl = context.http_controller();

  if (input.empty()) {
    // context.set_has_err(true);
    context.set_err_msg("Infer out is empty.");
    cntl->http_response().set_status_code(brpc::HTTP_STATUS_INTERNAL_SERVER_ERROR);
    cntl->http_response().set_content_type("text/plain");
    cntl->response_attachment().append(context.err_msg());
    CLOG4(ERROR, "YourConverter::PostProcess failed, err msg: " << context.err_msg());
    return;
  }

  // Build response json body.
  auto c = input[0].second.float_tensor;
  rapidjson::Document doc;
  doc.SetObject();
  doc.AddMember("c", c, doc.GetAllocator());
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  // Set response.
  cntl->http_response().set_status_code(brpc::HTTP_STATUS_OK);
  cntl->http_response().set_content_type("application/json");
  cntl->response_attachment().append(buffer.GetString());
}

// Implement and register other converter class after here.

} // namespace netease::grps
