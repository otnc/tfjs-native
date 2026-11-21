// Tensor conversion: JS TypedArray <-> TF_Tensor <-> TFE_TensorHandle (M1).
//
// A handle is exposed to JS as a Napi::External wrapping a HandleBox. The box
// lets deleteHandle() free the handle eagerly while the External's finalizer
// still runs safely at GC time (double-free is guarded by the null check).

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tensorflow/c/tf_tstring.h"
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

// TF's own deallocator for TF_AllocateTensor frees the buffer but never runs
// TF_StringDealloc, so strings too long for the inline small-string storage
// would leak. String tensors therefore get their own buffer and deallocator.
constexpr size_t kTensorAlignment = 64;

struct StringBuffer {
  void* raw;  // malloc'd block; the TF_TString array starts aligned inside it
};

void DeallocateStrings(void* data, size_t len, void* arg) {
  auto* strings = static_cast<TF_TString*>(data);
  const size_t count = len / sizeof(TF_TString);
  for (size_t i = 0; i < count; i++) {
    TF_StringDealloc(&strings[i]);
  }
  auto* buffer = static_cast<StringBuffer*>(arg);
  std::free(buffer->raw);
  delete buffer;
}

// createStringHandle(values: Uint8Array[], shape: number[]) -> External
Napi::Value CreateStringHandle(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  Napi::Array values = info[0].As<Napi::Array>();
  Napi::Array shapeArr = info[1].As<Napi::Array>();

  std::vector<int64_t> dims;
  dims.reserve(shapeArr.Length());
  for (uint32_t i = 0; i < shapeArr.Length(); i++) {
    dims.push_back(shapeArr.Get(i).As<Napi::Number>().Int64Value());
  }

  const size_t count = values.Length();
  const size_t byteLength = count * sizeof(TF_TString);
  void* raw = std::malloc(byteLength + kTensorAlignment);
  if (raw == nullptr) {
    Napi::Error::New(env, "tfjs-native: out of memory allocating a string tensor")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  const auto address = reinterpret_cast<uintptr_t>(raw);
  auto* strings =
      reinterpret_cast<TF_TString*>((address + kTensorAlignment - 1) & ~(kTensorAlignment - 1));
  for (uint32_t i = 0; i < count; i++) {
    Napi::Uint8Array bytes = values.Get(i).As<Napi::Uint8Array>();
    TF_StringInit(&strings[i]);
    TF_StringCopy(&strings[i], reinterpret_cast<const char*>(bytes.Data()), bytes.ByteLength());
  }

  // From here on TF owns the buffer: DeallocateStrings runs when the last
  // reference to it goes away, including on the error paths below.
  TF_Tensor* tensor = TF_NewTensor(TF_STRING, dims.data(), static_cast<int>(dims.size()), strings,
                                   byteLength, DeallocateStrings, new StringBuffer{raw});
  if (tensor == nullptr) {
    Napi::Error::New(env, "tfjs-native: TF_NewTensor failed for a string tensor")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

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

// handleStrings(h) -> Buffer[] (a copy of each element's bytes)
Napi::Value HandleStrings(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  TFE_TensorHandle* handle = UnwrapBox(info[0])->handle;

  TF_Status* status = TF_NewStatus();
  TF_Tensor* tensor = TFE_TensorHandleResolve(handle, status);
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }
  TF_DeleteStatus(status);

  if (TF_TensorType(tensor) != TF_STRING) {
    TF_DeleteTensor(tensor);
    Napi::Error::New(env, "tfjs-native: handleStrings called on a non-string tensor")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  const auto count = static_cast<size_t>(TF_TensorElementCount(tensor));
  const auto* strings = static_cast<const TF_TString*>(TF_TensorData(tensor));
  Napi::Array out = Napi::Array::New(env, count);
  for (size_t i = 0; i < count; i++) {
    const char* data = TF_StringGetDataPointer(&strings[i]);
    const size_t size = TF_StringGetSize(&strings[i]);
    out.Set(static_cast<uint32_t>(i),
            Napi::Buffer<uint8_t>::Copy(env, reinterpret_cast<const uint8_t*>(data), size));
  }

  TF_DeleteTensor(tensor);
  return out;
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
  exports.Set("createStringHandle", Napi::Function::New(env, CreateStringHandle));
  exports.Set("handleStrings", Napi::Function::New(env, HandleStrings));
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
