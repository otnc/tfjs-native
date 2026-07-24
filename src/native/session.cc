// Session execution shared by SavedModel inference (M3) and graphs built for
// gradients (M4). Feeds and fetches are addressed by graph tensor name + index.

#include <string>
#include <vector>

#include "tfjs_native.h"

namespace tfjs_native {

void FreeSession(SessionBox* box) {
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

Napi::External<SessionBox> WrapSession(Napi::Env env, TF_Session* session, TF_Graph* graph) {
  auto* box = new SessionBox{session, graph};
  return Napi::External<SessionBox>::New(env, box, [](Napi::Env, SessionBox* b) {
    FreeSession(b);
    delete b;
  });
}

SessionBox* UnwrapSession(const Napi::Value& value) {
  return value.As<Napi::External<SessionBox>>().Data();
}

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

namespace {

// sessionRun(session, inputOps, inputIndices, inputHandles, outputOps, outputIndices) -> Handle[]
Napi::Value SessionRun(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  SessionBox* box = UnwrapSession(info[0]);
  if (box->session == nullptr) {
    Napi::Error::New(env, "tfjs-native: session is already disposed").ThrowAsJavaScriptException();
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

    TF_Tensor* tensor = TFE_TensorHandleResolve(UnwrapBox(inputHandles.Get(i))->handle, status);
    if (ThrowIfError(env, status)) {
      cleanupInputs();
      TF_DeleteStatus(status);
      return env.Undefined();
    }
    inputValues.push_back(tensor);
  }

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

  // Hand results back as eager handles so they become normal Tensors.
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

// sessionDelete(session) -> void (idempotent)
Napi::Value SessionDelete(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  FreeSession(UnwrapSession(info[0]));
  return env.Undefined();
}

}  // namespace

void RegisterSession(Napi::Env env, Napi::Object exports) {
  exports.Set("sessionRun", Napi::Function::New(env, SessionRun));
  exports.Set("sessionDelete", Napi::Function::New(env, SessionDelete));
}

}  // namespace tfjs_native
