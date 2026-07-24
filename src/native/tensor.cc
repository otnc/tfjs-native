// Tensor conversion: JS TypedArray <-> TF_Tensor <-> TFE_TensorHandle (M1).
//
// A handle is exposed to JS as a Napi::External wrapping a HandleBox. The box
// lets deleteHandle() free the handle eagerly while the External's finalizer
// still runs safely at GC time (double-free is guarded by the null check).

#include <cstring>
#include <vector>

#include "tfjs_native.h"

namespace tfjs_native {

Napi::External<HandleBox> WrapHandle(Napi::Env env, TFE_TensorHandle* handle) {
  auto* box = new HandleBox{handle};
  return Napi::External<HandleBox>::New(env, box, [](Napi::Env, HandleBox* b) {
    if (b->handle != nullptr) {
      TFE_DeleteTensorHandle(b->handle);
    }
    delete b;
  });
}

HandleBox* UnwrapBox(const Napi::Value& value) {
  return value.As<Napi::External<HandleBox>>().Data();
}

namespace {

// createHandle(data: TypedArray, shape: number[], dtype: number) -> External
Napi::Value CreateHandle(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  Napi::TypedArray data = info[0].As<Napi::TypedArray>();
  Napi::Array shapeArr = info[1].As<Napi::Array>();
  auto dtype = static_cast<TF_DataType>(info[2].As<Napi::Number>().Int32Value());

  std::vector<int64_t> dims;
  dims.reserve(shapeArr.Length());
  for (uint32_t i = 0; i < shapeArr.Length(); i++) {
    dims.push_back(shapeArr.Get(i).As<Napi::Number>().Int64Value());
  }

  const size_t byteLength = data.ByteLength();
  const uint8_t* src = static_cast<uint8_t*>(data.ArrayBuffer().Data()) + data.ByteOffset();

  TF_Tensor* tensor =
      TF_AllocateTensor(dtype, dims.data(), static_cast<int>(dims.size()), byteLength);
  if (tensor == nullptr) {
    Napi::Error::New(env, "tfjs-native: TF_AllocateTensor failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  std::memcpy(TF_TensorData(tensor), src, byteLength);

  TF_Status* status = TF_NewStatus();
  TFE_TensorHandle* handle = TFE_NewTensorHandle(tensor, status);
  TF_DeleteTensor(tensor);
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }
  TF_DeleteStatus(status);

  return WrapHandle(env, handle);
}

// handleShape(h) -> number[]
Napi::Value HandleShape(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  TFE_TensorHandle* handle = UnwrapBox(info[0])->handle;

  TF_Status* status = TF_NewStatus();
  int numDims = TFE_TensorHandleNumDims(handle, status);
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }

  Napi::Array out = Napi::Array::New(env, numDims);
  for (int i = 0; i < numDims; i++) {
    int64_t dim = TFE_TensorHandleDim(handle, i, status);
    if (ThrowIfError(env, status)) {
      TF_DeleteStatus(status);
      return env.Undefined();
    }
    out.Set(static_cast<uint32_t>(i), Napi::Number::New(env, static_cast<double>(dim)));
  }
  TF_DeleteStatus(status);
  return out;
}

// handleDtype(h) -> number (TF_DataType)
Napi::Value HandleDtype(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  TFE_TensorHandle* handle = UnwrapBox(info[0])->handle;
  return Napi::Number::New(env, static_cast<int>(TFE_TensorHandleDataType(handle)));
}

// handleData(h) -> Buffer (copy of the tensor's raw bytes)
Napi::Value HandleData(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  TFE_TensorHandle* handle = UnwrapBox(info[0])->handle;

  TF_Status* status = TF_NewStatus();
  TF_Tensor* tensor = TFE_TensorHandleResolve(handle, status);
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }

  const size_t bytes = TF_TensorByteSize(tensor);
  Napi::Buffer<uint8_t> out =
      Napi::Buffer<uint8_t>::Copy(env, static_cast<uint8_t*>(TF_TensorData(tensor)), bytes);

  TF_DeleteTensor(tensor);
  TF_DeleteStatus(status);
  return out;
}

// deleteHandle(h) -> void (idempotent)
Napi::Value DeleteHandle(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  HandleBox* box = UnwrapBox(info[0]);
  if (box->handle != nullptr) {
    TFE_DeleteTensorHandle(box->handle);
    box->handle = nullptr;
  }
  return env.Undefined();
}

}  // namespace

void RegisterTensor(Napi::Env env, Napi::Object exports) {
  exports.Set("createHandle", Napi::Function::New(env, CreateHandle));
  exports.Set("handleShape", Napi::Function::New(env, HandleShape));
  exports.Set("handleDtype", Napi::Function::New(env, HandleDtype));
  exports.Set("handleData", Napi::Function::New(env, HandleData));
  exports.Set("deleteHandle", Napi::Function::New(env, DeleteHandle));
}

bool ThrowIfError(Napi::Env env, TF_Status* status) {
  if (TF_GetCode(status) == TF_OK) {
    return false;
  }
  Napi::Error err = Napi::Error::New(env, TF_Message(status));
  err.Set("code", Napi::Number::New(env, TF_GetCode(status)));
  err.ThrowAsJavaScriptException();
  return true;
}

}  // namespace tfjs_native
