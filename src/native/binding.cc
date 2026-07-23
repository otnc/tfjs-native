// N-API entry point for tfjs-native.
//
// This file wires the addon's exported functions. At milestone M0 it only
// exposes `version()` (a smoke test that libtensorflow is linked and loadable).
// Tensor conversion and eager execution live in tensor.cc / execute.cc and are
// registered here as the surface grows (see docs/DESIGN.md section 5).

#include <napi.h>

#include "tensorflow/c/c_api.h"
#include "tfjs_native.h"

namespace tfjs_native {

// Returns the linked libtensorflow version string, e.g. "2.x.y".
Napi::Value Version(const Napi::CallbackInfo& info) {
  return Napi::String::New(info.Env(), TF_Version());
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("version", Napi::Function::New(env, Version));
  RegisterTensor(env, exports);   // createHandle / handleData / deleteHandle ...
  RegisterExecute(env, exports);  // execute / getAllOps ...
  RegisterModel(env, exports);    // loadSavedModel / runSavedModel ...
  return exports;
}

NODE_API_MODULE(tfjs_native, Init)

}  // namespace tfjs_native
