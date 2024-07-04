/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2023/09/06
 * Brief  Brpc client demo. Complete interface description can be learned from docs/2_Interface.md.
 */

#include <brpc/channel.h>
#include <butil/logging.h>
#include <gflags/gflags.h>
#include <grps_apis/grps.brpc.pb.h>

#include <fstream>

DEFINE_string(server, "0.0.0.0:8021", "IP Address of server");
DEFINE_string(img_path, "", "Path of image");
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
  // Read image from img_path.
  if (FLAGS_img_path.empty()) {
    LOG(ERROR) << "Please input image path";
    return -1;
  }
  std::ifstream ifs(FLAGS_img_path, std::ios::binary);
  std::string image_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  request.set_bin_data(std::move(image_str));
  stub.Predict(&cntl, &request, &response, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "Fail to send predict request, " << cntl.ErrorText();
    return -1;
  }
  LOG(INFO) << "Predict label: " << response.str_data() << ", latency: " << cntl.latency_us() << " us";

  return 0;
}
