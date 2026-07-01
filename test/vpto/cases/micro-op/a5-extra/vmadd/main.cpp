// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                          \
  do {                                                                           \
    const aclError _ret = (expr);                                                \
    if (_ret != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,             \
                   (int)_ret, __FILE__, __LINE__);                               \
      const char *_recent = aclGetRecentErrMsg();                                \
      if (_recent != nullptr && _recent[0] != '\0')                              \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);             \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

#define FILE_CHECK(expr, path)                                                   \
  do {                                                                           \
    if (!(expr)) {                                                               \
      std::fprintf(stderr, "[ERROR] file operation failed: %s (%s:%d)\n",       \
                   path, __FILE__, __LINE__);                                    \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchA5ExtraVmadd(float *p0, float *p1, float *p2, float *p3,
                        void *stream);

struct Buffer {
  const char *path;
  size_t size;
  bool input;
  void *host;
  void *device;
};

int main() {
  constexpr size_t kElems = 1024;
  constexpr size_t kF32Bytes = kElems * sizeof(float);

  Buffer bufs[] = {
      {"./f_acc.bin", kF32Bytes, true, nullptr, nullptr},
      {"./f_lhs.bin", kF32Bytes, true, nullptr, nullptr},
      {"./f_rhs.bin", kF32Bytes, true, nullptr, nullptr},
      {"./out_vmadd.bin", kF32Bytes, false, nullptr, nullptr},
  };

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  for (Buffer &buf : bufs) {
    ACL_CHECK(aclrtMallocHost(&buf.host, buf.size));
    ACL_CHECK(aclrtMalloc(&buf.device, buf.size, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!buf.input)
      continue;
    size_t fileSize = buf.size;
    FILE_CHECK(ReadFile(buf.path, fileSize, buf.host, buf.size) &&
                   fileSize == buf.size,
               buf.path);
    ACL_CHECK(aclrtMemcpy(buf.device, buf.size, buf.host, buf.size,
                          ACL_MEMCPY_HOST_TO_DEVICE));
  }

  LaunchA5ExtraVmadd(
      static_cast<float *>(bufs[0].device), static_cast<float *>(bufs[1].device),
      static_cast<float *>(bufs[2].device),
      static_cast<float *>(bufs[3].device),
      stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));

  for (Buffer &buf : bufs) {
    if (buf.input)
      continue;
    ACL_CHECK(aclrtMemcpy(buf.host, buf.size, buf.device, buf.size,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    FILE_CHECK(WriteFile(buf.path, buf.host, buf.size), buf.path);
  }

cleanup:
  for (Buffer &buf : bufs) {
    if (buf.device != nullptr)
      aclrtFree(buf.device);
    if (buf.host != nullptr)
      aclrtFreeHost(buf.host);
  }
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
