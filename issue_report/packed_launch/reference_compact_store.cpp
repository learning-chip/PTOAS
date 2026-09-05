// TileLang-free ASC/CCE reference for desired_compact_store.py.
#include "reference_compact_store.asc"

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" void launch_reference_compact_f32_store(
    void *stream, void *input, void *output) {
  reference_compact_f32_store<<<1, nullptr, stream>>>(
      (__gm__ float *)input, (__gm__ float *)output);
}
