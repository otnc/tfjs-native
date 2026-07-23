// Eager execution and op-registry access.
//
// execute() is the single door for running any op: it builds a TFE_Op from an op
// name + input handles + attributes, runs it on the process-wide eager context,
// and returns the output handles. getAllOps() feeds the codegen pipeline.

#include <string>
#include <vector>

#include "tfjs_native.h"

namespace tfjs_native {

TFE_Context* GetContext(Napi::Env env) {
  static TFE_Context* ctx = nullptr;
  if (ctx == nullptr) {
    TF_Status* status = TF_NewStatus();
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    ctx = TFE_NewContext(opts, status);
    TFE_DeleteContextOptions(opts);
    if (TF_GetCode(status) != TF_OK) {
      Napi::Error::New(env, TF_Message(status)).ThrowAsJavaScriptException();
      TF_DeleteStatus(status);
      ctx = nullptr;
      return nullptr;
    }
    TF_DeleteStatus(status);
  }
  return ctx;
}

namespace {

// Applies a JS attrs object onto `op`. Each entry is `name -> {type, value}`.
// Returns false (with a JS error thrown) on an unsupported type or TF error.
bool SetAttrs(Napi::Env env, TFE_Op* op, Napi::Object attrs, TF_Status* status) {
  Napi::Array names = attrs.GetPropertyNames();
  for (uint32_t i = 0; i < names.Length(); i++) {
    std::string name = names.Get(i).As<Napi::String>().Utf8Value();
    Napi::Object spec = attrs.Get(name).As<Napi::Object>();
    std::string type = spec.Get("type").As<Napi::String>().Utf8Value();
    Napi::Value v = spec.Get("value");
    const char* n = name.c_str();

    if (type == "type") {
      TFE_OpSetAttrType(op, n, static_cast<TF_DataType>(v.As<Napi::Number>().Int32Value()));
    } else if (type == "bool") {
      TFE_OpSetAttrBool(op, n, v.As<Napi::Boolean>().Value() ? 1 : 0);
    } else if (type == "int") {
      TFE_OpSetAttrInt(op, n, v.As<Napi::Number>().Int64Value());
    } else if (type == "float") {
      TFE_OpSetAttrFloat(op, n, v.As<Napi::Number>().FloatValue());
    } else if (type == "string") {
      std::string s = v.As<Napi::String>().Utf8Value();
      TFE_OpSetAttrString(op, n, s.data(), s.size());
    } else if (type == "shape") {
      Napi::Array arr = v.As<Napi::Array>();
      std::vector<int64_t> dims;
      for (uint32_t j = 0; j < arr.Length(); j++) {
        dims.push_back(arr.Get(j).As<Napi::Number>().Int64Value());
      }
      TFE_OpSetAttrShape(op, n, dims.data(), static_cast<int>(dims.size()), status);
      if (ThrowIfError(env, status)) return false;
    } else if (type == "typeList") {
      Napi::Array arr = v.As<Napi::Array>();
      std::vector<TF_DataType> vals;
      for (uint32_t j = 0; j < arr.Length(); j++) {
        vals.push_back(static_cast<TF_DataType>(arr.Get(j).As<Napi::Number>().Int32Value()));
      }
      TFE_OpSetAttrTypeList(op, n, vals.data(), static_cast<int>(vals.size()));
    } else if (type == "intList") {
      Napi::Array arr = v.As<Napi::Array>();
      std::vector<int64_t> vals;
      for (uint32_t j = 0; j < arr.Length(); j++) {
        vals.push_back(arr.Get(j).As<Napi::Number>().Int64Value());
      }
      TFE_OpSetAttrIntList(op, n, vals.data(), static_cast<int>(vals.size()));
    } else if (type == "floatList") {
      Napi::Array arr = v.As<Napi::Array>();
      std::vector<float> vals;
      for (uint32_t j = 0; j < arr.Length(); j++) {
        vals.push_back(arr.Get(j).As<Napi::Number>().FloatValue());
      }
      TFE_OpSetAttrFloatList(op, n, vals.data(), static_cast<int>(vals.size()));
    } else if (type == "boolList") {
      Napi::Array arr = v.As<Napi::Array>();
      std::vector<unsigned char> vals;
      for (uint32_t j = 0; j < arr.Length(); j++) {
        vals.push_back(arr.Get(j).As<Napi::Boolean>().Value() ? 1 : 0);
      }
      TFE_OpSetAttrBoolList(op, n, vals.data(), static_cast<int>(vals.size()));
    } else if (type == "stringList") {
      Napi::Array arr = v.As<Napi::Array>();
      std::vector<std::string> storage;
      for (uint32_t j = 0; j < arr.Length(); j++) {
        storage.push_back(arr.Get(j).As<Napi::String>().Utf8Value());
      }
      std::vector<const void*> ptrs;
      std::vector<size_t> lens;
      for (const std::string& s : storage) {
        ptrs.push_back(s.data());
        lens.push_back(s.size());
      }
      TFE_OpSetAttrStringList(op, n, ptrs.data(), lens.data(), static_cast<int>(storage.size()));
    } else {
      Napi::Error::New(env, "tfjs-native: unsupported attr type '" + type + "' for '" + name + "'")
          .ThrowAsJavaScriptException();
      return false;
    }
  }
  return true;
}

// execute(opName: string, inputs: Handle[], attrs: object, numOutputs: number) -> Handle[]
Napi::Value Execute(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  std::string opName = info[0].As<Napi::String>().Utf8Value();
  Napi::Array inputs = info[1].As<Napi::Array>();
  Napi::Object attrs = info[2].As<Napi::Object>();
  int numOutputs = info[3].As<Napi::Number>().Int32Value();

  TFE_Context* ctx = GetContext(env);
  if (ctx == nullptr) return env.Undefined();

  TF_Status* status = TF_NewStatus();
  TFE_Op* op = TFE_NewOp(ctx, opName.c_str(), status);
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }

  for (uint32_t i = 0; i < inputs.Length(); i++) {
    TFE_OpAddInput(op, UnwrapBox(inputs.Get(i))->handle, status);
    if (ThrowIfError(env, status)) {
      TFE_DeleteOp(op);
      TF_DeleteStatus(status);
      return env.Undefined();
    }
  }

  if (!SetAttrs(env, op, attrs, status)) {
    TFE_DeleteOp(op);
    TF_DeleteStatus(status);
    return env.Undefined();
  }

  std::vector<TFE_TensorHandle*> retvals(static_cast<size_t>(numOutputs), nullptr);
  int actual = numOutputs;
  TFE_Execute(op, retvals.data(), &actual, status);
  TFE_DeleteOp(op);
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }
  TF_DeleteStatus(status);

  Napi::Array out = Napi::Array::New(env, static_cast<size_t>(actual));
  for (int i = 0; i < actual; i++) {
    out.Set(static_cast<uint32_t>(i), WrapHandle(env, retvals[i]));
  }
  return out;
}

// getAllOps() -> Buffer of the serialized OpList proto (for codegen).
Napi::Value GetAllOps(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  TF_Buffer* buf = TF_GetAllOpList();
  Napi::Buffer<uint8_t> out =
      Napi::Buffer<uint8_t>::Copy(env, static_cast<const uint8_t*>(buf->data), buf->length);
  TF_DeleteBuffer(buf);
  return out;
}

}  // namespace

void RegisterExecute(Napi::Env env, Napi::Object exports) {
  exports.Set("execute", Napi::Function::New(env, Execute));
  exports.Set("getAllOps", Napi::Function::New(env, GetAllOps));
  // TODO(M4): gradient(ctx, ys, xs, dys).
}

}  // namespace tfjs_native
