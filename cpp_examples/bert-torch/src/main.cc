/*
 * Copyright 2022 netease. All rights reserved.
 * Author zhaochaochao@corp.netease.com
 * Date   2023/02/06
 * Brief  Unit test main. Test model inferer and converter here.
 */
#include <gtest/gtest.h>

#include <memory>

#include "executor/executor.h"
#include "grps_server_customized.h"
#include "logger/logger.h"
#include "monitor/monitor.h"
#include "system_monitor/system_monitor.h"

using namespace netease::grps;

TEST(LocalTest, TestInit) {
  // Load global config.
  ASSERT_TRUE(GlobalConfig::Instance().Load());

  // Init logger.
  const auto& server_config = GlobalConfig::Instance().server_config();
  std::string sys_log_path = server_config.log.log_dir + "/grps_server.log";
  std::string usr_log_path = server_config.log.log_dir + "/grps_usr.log";
  ASSERT_NO_THROW(DailyLogger::Instance().Init(sys_log_path, server_config.log.log_backup_count, usr_log_path,
                                               server_config.log.log_backup_count));

  // Init system monitor.
  ASSERT_NO_THROW(Monitor::Instance().Init());
  ASSERT_NO_THROW(SystemMonitor::Instance().Init());

  // Init executor.
  ASSERT_NO_THROW(Executor::Instance().Init());
}

TEST(LocalTest, TestInfer) {
  ::grps::protos::v1::GrpsMessage inp_grps_message;
  inp_grps_message.set_str_data("中国的首都是哪里？ [SEP] 北京是 [MASK] 国的首都。");
  // Add codes to set input message here.
  ::grps::protos::v1::GrpsMessage out_grps_message;
  auto ctx_sp = std::make_shared<GrpsContext>(&inp_grps_message);
  ASSERT_NO_THROW(Executor::Instance().Infer(inp_grps_message, out_grps_message, ctx_sp));
  ASSERT_FALSE(ctx_sp->has_err());
  // Add codes to check output message here.
  ASSERT_EQ(out_grps_message.str_data(), "中");
}

int main(int argc, char** argv) {
  // Init grps-server-customized lib.
  GrpsServerCustomizedLibInit();

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  // Terminate executor.
  Executor::Instance().Terminate();
  return ret;
}
