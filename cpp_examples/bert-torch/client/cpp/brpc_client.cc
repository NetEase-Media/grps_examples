/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2023/09/06
 * Brief  Brpc client demo. Complete interface description can be learned from docs/2_Interface.md.
 */

#include <brpc/channel.h>
#include <butil/logging.h>
#include <gflags/gflags.h>
#include <google/protobuf/text_format.h>
#include <grps_apis/grps.brpc.pb.h>

#include <fstream>

DEFINE_string(server, "0.0.0.0:8021", "IP Address of server");
DEFINE_string(inp, "", "input");
DEFINE_string(load_balancer, "", "The algorithm for load balancing");
DEFINE_string(connection_type, "", "Connection type. Available values: single, pooled, short");
DEFINE_int32(timeout_ms, 400, "RPC timeout in milliseconds");
DEFINE_int32(max_retry, 3, "Max retries(not including the first RPC)");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  brpc::Channel channel;

  brpc::ChannelOptions options;
  options.protocol = brpc::PROTOCOL_BAIDU_STD;
  options.connection_type = FLAGS_connection_type;
  options.timeout_ms = FLAGS_timeout_ms /*milliseconds*/;
  options.max_retry = FLAGS_max_retry;

  if (channel.Init(FLAGS_server.c_str(), FLAGS_load_balancer.c_str(), &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel";
    return -1;
  }

  grps::protos::v1::GrpsBrpcService_Stub stub(&channel);

  grps::protos::v1::GrpsMessage request;
  grps::protos::v1::GrpsMessage response;

  brpc::Controller cntl;

  // Predict request.
  request.set_str_data(FLAGS_inp);
  stub.Predict(&cntl, &request, &response, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to send predict request, " << cntl.ErrorText();
    return -1;
  }
  std::string res_str;
  ::google::protobuf::TextFormat::PrintToString(response, &res_str);
  LOG(INFO) << "Predict response: " << res_str << ", decoded str_data: " << response.str_data()
            << ", latency: " << cntl.latency_us() << " us";

  return 0;
}
