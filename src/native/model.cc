// SavedModel loading (M3).
//
// TF_LoadSessionFromSavedModel gives us a TF_Session plus the graph it was
// restored into, and the serialized MetaGraphDef. Running and disposing are
// shared with graph-built sessions (see session.cc); signature resolution
// happens on the TS side, so this layer only loads.

#include <string>
#include <vector>

#include "tfjs_native.h"

namespace tfjs_native {

namespace {

// loadSavedModel(dir: string, tags: string[]) -> { handle, metaGraphDef }
Napi::Value LoadSavedModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  std::string dir = info[0].As<Napi::String>().Utf8Value();
  Napi::Array tagArr = info[1].As<Napi::Array>();

  std::vector<std::string> tagStorage;
  tagStorage.reserve(tagArr.Length());
  for (uint32_t i = 0; i < tagArr.Length(); i++) {
    tagStorage.push_back(tagArr.Get(i).As<Napi::String>().Utf8Value());
  }
  std::vector<const char*> tags;
  tags.reserve(tagStorage.size());
  for (const std::string& t : tagStorage) {
    tags.push_back(t.c_str());
  }

  TF_Status* status = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();
  TF_Buffer* metaGraphDef = TF_NewBuffer();
  TF_SessionOptions* opts = TF_NewSessionOptions();

  TF_Session* session =
      TF_LoadSessionFromSavedModel(opts, nullptr, dir.c_str(), tags.data(),
                                   static_cast<int>(tags.size()), graph, metaGraphDef, status);
  TF_DeleteSessionOptions(opts);

  if (ThrowIfError(env, status)) {
    TF_DeleteBuffer(metaGraphDef);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    return env.Undefined();
  }
  TF_DeleteStatus(status);

  Napi::Buffer<uint8_t> meta = Napi::Buffer<uint8_t>::Copy(
      env, static_cast<const uint8_t*>(metaGraphDef->data), metaGraphDef->length);
  TF_DeleteBuffer(metaGraphDef);

  Napi::Object result = Napi::Object::New(env);
  result.Set("handle", WrapSession(env, session, graph));
  result.Set("metaGraphDef", meta);
  return result;
}

}  // namespace

void RegisterModel(Napi::Env env, Napi::Object exports) {
  exports.Set("loadSavedModel", Napi::Function::New(env, LoadSavedModel));
}

}  // namespace tfjs_native
