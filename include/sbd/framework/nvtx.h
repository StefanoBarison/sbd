/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SBD_FRAMEWORK_NVTX_H_
#define SBD_FRAMEWORK_NVTX_H_

#include <cstdint>
#include <string>

#if defined(SBD_USE_NVTX) && defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define SBD_NVTX_ENABLED 1
#  else
#    define SBD_NVTX_ENABLED 0
#  endif
#else
#  define SBD_NVTX_ENABLED 0
#endif

namespace sbd {
namespace framework {

inline const char* nvtx_basename(const char* path) {
  const char* base = path;
  for (const char* p = path; *p != '\0'; ++p) {
    if (*p == '/' || *p == '\\') base = p + 1;
  }
  return base;
}

inline std::string nvtx_make_label(const char* file, const char* func, int line) {
  std::string s;
  s.reserve(64);
  s += nvtx_basename(file);
  s += ":";
  s += std::to_string(line);
  s += " [";
  s += func;
  s += "]";
  return s;
}

inline std::string nvtx_make_label(const char* file, const char* func, int line,
                                   const char* msg) {
  std::string s = nvtx_make_label(file, func, line);
  if (msg && *msg) {
    s += " : ";
    s += msg;
  }
  return s;
}

inline std::uint32_t nvtx_set_color(std::uint32_t color_id)
{
    std::uint32_t color_r = (((color_id & 0xFF) * 37) & 0xFF) << 16;
    std::uint32_t color_g = (((color_id & 0xFF) * 67) & 0xFF) <<  8;
    std::uint32_t color_b = (((color_id & 0xFF) * 97) & 0xFF);
    return 0xFF000000 | color_r | color_g | color_b;
}


class NvtxRange {
 public:
  NvtxRange(const char* file, const char* func, int line,
            std::uint32_t color_id = 0)
      : label_(nvtx_make_label(file, func, line)) {
#if SBD_NVTX_ENABLED
    nvtxEventAttributes_t ev{};
    ev.version = NVTX_VERSION;
    ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    ev.colorType = NVTX_COLOR_ARGB;
    ev.color = nvtx_set_color(color_id);
    ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
    ev.message.ascii = label_.c_str();
    nvtxRangePushEx(&ev);
#else
    (void)color_id;
#endif
  }

  NvtxRange(const char* file, const char* func, int line,
            const char* msg,
            std::uint32_t color_id = 0)
      : label_(nvtx_make_label(file, func, line, msg)) {
#if SBD_NVTX_ENABLED
    nvtxEventAttributes_t ev{};
    ev.version = NVTX_VERSION;
    ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    ev.colorType = NVTX_COLOR_ARGB;
    ev.color = nvtx_set_color(color_id);
    ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
    ev.message.ascii = label_.c_str();
    nvtxRangePushEx(&ev);
#else
    (void)color_id;
#endif
  }

  ~NvtxRange() {
#if SBD_NVTX_ENABLED
    nvtxRangePop();
#endif
  }

  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;

 private:
  std::string label_;
};

inline void nvtx_mark(const char* file, const char* func, int line) {
#if SBD_NVTX_ENABLED
  const std::string label = nvtx_make_label(file, func, line);

  nvtxEventAttributes_t ev{};
  ev.version = NVTX_VERSION;
  ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
  ev.message.ascii = label.c_str();
  nvtxMarkEx(&ev);
#else
  (void)file;
  (void)func;
  (void)line;
#endif
}

inline void nvtx_mark(const char* file, const char* func, int line,
                      const char* msg) {
#if SBD_NVTX_ENABLED
  const std::string label = nvtx_make_label(file, func, line, msg);

  nvtxEventAttributes_t ev{};
  ev.version = NVTX_VERSION;
  ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
  ev.message.ascii = label.c_str();
  nvtxMarkEx(&ev);
#else
  (void)file;
  (void)func;
  (void)line;
  (void)msg;
#endif
}

}  // namespace framework
}  // namespace sbd

#define SBD_NVTX_CONCAT_IMPL(x, y) x##y
#define SBD_NVTX_CONCAT(x, y) SBD_NVTX_CONCAT_IMPL(x, y)

#define SBD_NVTX_RANGE_AUTO()                                                   \
  ::sbd::framework::NvtxRange SBD_NVTX_CONCAT(_sbd_nvtx_range_, __LINE__)(      \
      __FILE__, __func__, __LINE__)

#define SBD_NVTX_RANGE_AUTO_COLOR(color)                                        \
  ::sbd::framework::NvtxRange SBD_NVTX_CONCAT(_sbd_nvtx_range_, __LINE__)(      \
      __FILE__, __func__, __LINE__, (color))

#define SBD_NVTX_RANGE(msg)                                                     \
  ::sbd::framework::NvtxRange SBD_NVTX_CONCAT(_sbd_nvtx_range_, __LINE__)(      \
      __FILE__, __func__, __LINE__, (msg))

#define SBD_NVTX_RANGE_COLOR(msg, color)                                        \
  ::sbd::framework::NvtxRange SBD_NVTX_CONCAT(_sbd_nvtx_range_, __LINE__)(      \
      __FILE__, __func__, __LINE__, (msg), (color))

#define SBD_NVTX_MARK_AUTO()                                                    \
  ::sbd::framework::nvtx_mark(__FILE__, __func__, __LINE__)

#define SBD_NVTX_MARK(msg)                                                      \
  ::sbd::framework::nvtx_mark(__FILE__, __func__, __LINE__, (msg))

#endif  // SBD_FRAMEWORK_NVTX_H_
