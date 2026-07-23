// SavedModel loading and session execution (M3).
//
// TF_LoadSessionFromSavedModel gives us a TF_Session plus the graph it was
// restored into, and the serialized MetaGraphDef. Signature resolution happens
// on the TS side (protobufjs), so this layer only deals with tensor names.

#include <string>
#include <vector>

#include "tfjs_native.h"

namespace tfjs_native {

namespace {

// Owns the session + graph so both are freed exactly once.
struct ModelBox {
  TF_Session* session;
  TF_Graph* graph;
};

void FreeModel(ModelBox* box) {
  if (box->session != nullptr) {
    TF_Status* status = TF_NewStatus();
    TF_CloseSession(box->session, status);
    TF_DeleteSession(box->session, status);
    TF_DeleteStatus(status);
    box->session = nullptr;
  }
  if (box->graph != nullptr) {
    TF_DeleteGraph(box->graph);
    box->graph = nullptr;
  }
}

ModelBox* UnwrapModel(const Napi::Value& value) {
  return value.As<Napi::External<ModelBox>>().Data();
}

// Resolves "opName" + index into a TF_Output, throwing when the op is missing.
bool ResolveOutput(Napi::Env env, TF_Graph* graph, const std::string& opName, int index,
                   TF_Output* out) {
  TF_Operation* oper = TF_GraphOperationByName(graph, opName.c_str());
  if (oper == nullptr) {
    Napi::Error::New(env, "tfjs-native: no operation named '" + opName + "' in the graph")
        .ThrowAsJavaScriptException();
    return false;
  }
  out->oper = oper;
  out->index = index;
  return true;
}

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

  auto* box = new ModelBox{session, graph};
  Napi::External<ModelBox> handle =
      Napi::External<ModelBox>::New(env, box, [](Napi::Env, ModelBox* b) {
        FreeModel(b);
        delete b;
      });

  Napi::Object result = Napi::Object::New(env);
  result.Set("handle", handle);
  result.Set("metaGraphDef", meta);
  return result;
}

// runSavedModel(handle, inputOps, inputIndices, inputHandles, outputOps, outputIndices) -> Handle[]
Napi::Value RunSavedModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ModelBox* box = UnwrapModel(info[0]);
  if (box->session == nullptr) {
    Napi::Error::New(env, "tfjs-native: SavedModel is already disposed")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  Napi::Array inputOps = info[1].As<Napi::Array>();
  Napi::Array inputIdx = info[2].As<Napi::Array>();
  Napi::Array inputHandles = info[3].As<Napi::Array>();
  Napi::Array outputOps = info[4].As<Napi::Array>();
  Napi::Array outputIdx = info[5].As<Napi::Array>();

  TF_Status* status = TF_NewStatus();

  // Feeds: resolve names and materialize each eager handle as a TF_Tensor.
  std::vector<TF_Output> inputs;
  std::vector<TF_Tensor*> inputValues;
  auto cleanupInputs = [&inputValues]() {
    for (TF_Tensor* t : inputValues) {
      if (t != nullptr) TF_DeleteTensor(t);
    }
  };
  for (uint32_t i = 0; i < inputOps.Length(); i++) {
    TF_Output out;
    std::string name = inputOps.Get(i).As<Napi::String>().Utf8Value();
    if (!ResolveOutput(env, box->graph, name, inputIdx.Get(i).As<Napi::Number>().Int32Value(),
                       &out)) {
      cleanupInputs();
      TF_DeleteStatus(status);
      return env.Undefined();
    }
    inputs.push_back(out);

    TFE_TensorHandle* handle = UnwrapBox(inputHandles.Get(i))->handle;
    TF_Tensor* tensor = TFE_TensorHandleResolve(handle, status);
    if (ThrowIfError(env, status)) {
      cleanupInputs();
      TF_DeleteStatus(status);
      return env.Undefined();
    }
    inputValues.push_back(tensor);
  }

  // Fetches
  std::vector<TF_Output> outputs;
  for (uint32_t i = 0; i < outputOps.Length(); i++) {
    TF_Output out;
    std::string name = outputOps.Get(i).As<Napi::String>().Utf8Value();
    if (!ResolveOutput(env, box->graph, name, outputIdx.Get(i).As<Napi::Number>().Int32Value(),
                       &out)) {
      cleanupInputs();
      TF_DeleteStatus(status);
      return env.Undefined();
    }
    outputs.push_back(out);
  }

  std::vector<TF_Tensor*> outputValues(outputs.size(), nullptr);
  TF_SessionRun(box->session, nullptr, inputs.data(), inputValues.data(),
                static_cast<int>(inputs.size()), outputs.data(), outputValues.data(),
                static_cast<int>(outputs.size()), nullptr, 0, nullptr, status);
  cleanupInputs();
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }

  // Hand the results back as eager handles so they become normal Tensors.
  Napi::Array result = Napi::Array::New(env, outputValues.size());
  for (size_t i = 0; i < outputValues.size(); i++) {
    TFE_TensorHandle* handle = TFE_NewTensorHandle(outputValues[i], status);
    TF_DeleteTensor(outputValues[i]);
    if (ThrowIfError(env, status)) {
      TF_DeleteStatus(status);
      return env.Undefined();
    }
    result.Set(static_cast<uint32_t>(i), WrapHandle(env, handle));
  }
  TF_DeleteStatus(status);
  return result;
}

// deleteSavedModel(handle) -> void (idempotent)
Napi::Value DeleteSavedModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  FreeModel(UnwrapModel(info[0]));
  return env.Undefined();
}

}  // namespace

void RegisterModel(Napi::Env env, Napi::Object exports) {
  exports.Set("loadSavedModel", Napi::Function::New(env, LoadSavedModel));
  exports.Set("runSavedModel", Napi::Function::New(env, RunSavedModel));
  exports.Set("deleteSavedModel", Napi::Function::New(env, DeleteSavedModel));
}

}  // namespace tfjs_native
