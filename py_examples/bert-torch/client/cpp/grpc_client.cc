/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2023/09/06
 * Brief  Grpc client demo. Complete interface description can be learned from docs/2_Interface.md.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <grpcpp/grpcpp.h>
#include <grps_apis/grps.grpc.pb.h>

#include <string>

DEFINE_string(server, "0.0.0.0:8081", "IP Address of server");
DEFINE_string(inp, "", "input");

#define GET_US() \
  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(FLAGS_server, grpc::InsecureChannelCredentials());

  std::unique_ptr<::grps::protos::v1::GrpsService::Stub> stub = ::grps::protos::v1::GrpsService::NewStub(channel);

  ::grps::protos::v1::GrpsMessage request;
  ::grps::protos::v1::GrpsMessage response;

  grpc::ClientContext context;
  request.set_str_data(FLAGS_inp);
  auto begin_us = GET_US();
  grpc::Status status = stub->Predict(&context, request, &response);
  if (!status.ok()) {
    LOG(ERROR) << "Fail to send predict request, " << status.error_message();
    return -1;
  }
  auto end_us = GET_US();
  std::string res_str;
  ::google::protobuf::TextFormat::PrintToString(response, &res_str);
  LOG(INFO) << "Predict response: " << res_str << ", decoded str_data: " << response.str_data()
            << ", latency: " << end_us - begin_us << " us";
  return 0;
}