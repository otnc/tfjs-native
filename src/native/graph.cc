// Graph construction (M4).
//
// The training path records eager ops and replays them into a TF_Graph so
// TF_AddGradients can differentiate them (the C API has no eager tape). Nodes
// are addressed as {op, index} ports, matching session.cc's feed/fetch naming.

#include <string>
#include <vector>

#include "tfjs_native.h"

namespace tfjs_native {

namespace {

// Owns a graph until a session takes it over (see GraphNewSession).
struct GraphBox {
  TF_Graph* graph;
  bool owned;
};

GraphBox* UnwrapGraph(const Napi::Value& value) {
  return value.As<Napi::External<GraphBox>>().Data();
}

Napi::Object MakePort(Napi::Env env, const std::string& op, int index) {
  Napi::Object port = Napi::Object::New(env);
  port.Set("op", Napi::String::New(env, op));
  port.Set("index", Napi::Number::New(env, index));
  return port;
}

bool PortFrom(Napi::Env env, TF_Graph* graph, const Napi::Value& value, TF_Output* out) {
  Napi::Object port = value.As<Napi::Object>();
  return ResolveOutput(env, graph, port.Get("op").As<Napi::String>().Utf8Value(),
                       port.Get("index").As<Napi::Number>().Int32Value(), out);
}

// Mirrors SetAttrs() in execute.cc, but against TF_OperationDescription. The
// two C APIs are parallel except that TF_SetAttrShape takes no status.
bool SetGraphAttrs(Napi::Env env, TF_OperationDescription* desc, Napi::Object attrs,
                   TF_Status* status) {
  Napi::Array names = attrs.GetPropertyNames();
  for (uint32_t i = 0; i < names.Length(); i++) {
    std::string name = names.Get(i).As<Napi::String>().Utf8Value();
    Napi::Object spec = attrs.Get(name).As<Napi::Object>();
    std::string type = spec.Get("type").As<Napi::String>().Utf8Value();
    Napi::Value v = spec.Get("value");
    const char* n = name.c_str();

    if (type == "type") {
      TF_SetAttrType(desc, n, static_cast<TF_DataType>(v.As<Napi::Number>().Int32Value()));
    } else if (type == "bool") {
      TF_SetAttrBool(desc, n, v.As<Napi::Boolean>().Value() ? 1 : 0);
    } else if (type == "int") {
      TF_SetAttrInt(desc, n, v.As<Napi::Number>().Int64Value());
    } else if (type == "float") {
      TF_SetAttrFloat(desc, n, v.As<Napi::Number>().FloatValue());
    } else if (type == "string") {
      std::string s = v.As<Napi::String>().Utf8Value();
      TF_SetAttrString(desc, n, s.data(), s.size());
    } else if (type == "shape") {
      Napi::Array arr = v.As<Napi::Array>();
      std::vector<int64_t> dims;
      for (uint32_t j = 0; j < arr.Length(); j++) {
        dims.push_back(arr.Get(j).As<Napi::Number>().Int64Value());
      }
      TF_SetAttrShape(desc, n, dims.data(), static_cast<int>(dims.size()));
    } else if (type == "typeList") {
      Napi::Array arr = v.As<Napi::Array>();
      std::vector<TF_DataType> vals;
      for (uint32_t j = 0; j < arr.Length(); j++) {
        vals.push_back(static_cast<TF_DataType>(arr.Get(j).As<Napi::Number>().Int32Value()));
      }
      TF_SetAttrTypeList(desc, n, vals.data(), static_cast<int>(vals.size()));
    } else if (type == "intList") {
      Napi::Array arr = v.As<Napi::Array>();
      std::vector<int64_t> vals;
      for (uint32_t j = 0; j < arr.Length(); j++) {
        vals.push_back(arr.Get(j).As<Napi::Number>().Int64Value());
      }
      TF_SetAttrIntList(desc, n, vals.data(), static_cast<int>(vals.size()));
    } else if (type == "floatList") {
      Napi::Array arr = v.As<Napi::Array>();
      std::vector<float> vals;
      for (uint32_t j = 0; j < arr.Length(); j++) {
        vals.push_back(arr.Get(j).As<Napi::Number>().FloatValue());
      }
      TF_SetAttrFloatList(desc, n, vals.data(), static_cast<int>(vals.size()));
    } else if (type == "boolList") {
      Napi::Array arr = v.As<Napi::Array>();
      std::vector<unsigned char> vals;
      for (uint32_t j = 0; j < arr.Length(); j++) {
        vals.push_back(arr.Get(j).As<Napi::Boolean>().Value() ? 1 : 0);
      }
      TF_SetAttrBoolList(desc, n, vals.data(), static_cast<int>(vals.size()));
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
      TF_SetAttrStringList(desc, n, ptrs.data(), lens.data(), static_cast<int>(storage.size()));
    } else {
      Napi::Error::New(env, "tfjs-native: unsupported attr type '" + type + "' for '" + name + "'")
          .ThrowAsJavaScriptException();
      return false;
    }
  }
  (void)status;
  return true;
}

// graphCreate() -> GraphHandle
Napi::Value GraphCreate(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* box = new GraphBox{TF_NewGraph(), true};
  return Napi::External<GraphBox>::New(env, box, [](Napi::Env, GraphBox* b) {
    if (b->owned && b->graph != nullptr) TF_DeleteGraph(b->graph);
    delete b;
  });
}

// graphPlaceholder(graph, name, dtype, shape | null) -> Port
Napi::Value GraphPlaceholder(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  GraphBox* box = UnwrapGraph(info[0]);
  std::string name = info[1].As<Napi::String>().Utf8Value();
  auto dtype = static_cast<TF_DataType>(info[2].As<Napi::Number>().Int32Value());

  TF_OperationDescription* desc = TF_NewOperation(box->graph, "Placeholder", name.c_str());
  TF_SetAttrType(desc, "dtype", dtype);
  if (info[3].IsArray()) {
    Napi::Array arr = info[3].As<Napi::Array>();
    std::vector<int64_t> dims;
    for (uint32_t i = 0; i < arr.Length(); i++) {
      dims.push_back(arr.Get(i).As<Napi::Number>().Int64Value());
    }
    TF_SetAttrShape(desc, "shape", dims.data(), static_cast<int>(dims.size()));
  } else {
    TF_SetAttrShape(desc, "shape", nullptr, -1);  // unknown rank
  }

  TF_Status* status = TF_NewStatus();
  TF_FinishOperation(desc, status);
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }
  TF_DeleteStatus(status);
  return MakePort(env, name, 0);
}

// graphConst(graph, name, handle) -> Port  (bakes an eager tensor into the graph)
Napi::Value GraphConst(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  GraphBox* box = UnwrapGraph(info[0]);
  std::string name = info[1].As<Napi::String>().Utf8Value();

  TF_Status* status = TF_NewStatus();
  TF_Tensor* tensor = TFE_TensorHandleResolve(UnwrapBox(info[2])->handle, status);
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }

  TF_OperationDescription* desc = TF_NewOperation(box->graph, "Const", name.c_str());
  TF_SetAttrType(desc, "dtype", TF_TensorType(tensor));
  TF_SetAttrTensor(desc, "value", tensor, status);
  TF_DeleteTensor(tensor);
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }
  TF_FinishOperation(desc, status);
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }
  TF_DeleteStatus(status);
  return MakePort(env, name, 0);
}

// graphAddOp(graph, opType, name, inputs, attrs, numOutputs) -> Port[]
// An input element may be a Port, or an array of Ports for a list-typed argument.
Napi::Value GraphAddOp(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  GraphBox* box = UnwrapGraph(info[0]);
  std::string opType = info[1].As<Napi::String>().Utf8Value();
  std::string name = info[2].As<Napi::String>().Utf8Value();
  Napi::Array inputs = info[3].As<Napi::Array>();
  Napi::Object attrs = info[4].As<Napi::Object>();
  int numOutputs = info[5].As<Napi::Number>().Int32Value();

  TF_OperationDescription* desc = TF_NewOperation(box->graph, opType.c_str(), name.c_str());
  for (uint32_t i = 0; i < inputs.Length(); i++) {
    Napi::Value element = inputs.Get(i);
    if (element.IsArray()) {
      Napi::Array list = element.As<Napi::Array>();
      std::vector<TF_Output> ports(list.Length());
      for (uint32_t j = 0; j < list.Length(); j++) {
        if (!PortFrom(env, box->graph, list.Get(j), &ports[j])) return env.Undefined();
      }
      TF_AddInputList(desc, ports.data(), static_cast<int>(ports.size()));
    } else {
      TF_Output port;
      if (!PortFrom(env, box->graph, element, &port)) return env.Undefined();
      TF_AddInput(desc, port);
    }
  }

  TF_Status* status = TF_NewStatus();
  if (!SetGraphAttrs(env, desc, attrs, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }
  TF_FinishOperation(desc, status);
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }
  TF_DeleteStatus(status);

  Napi::Array out = Napi::Array::New(env, static_cast<size_t>(numOutputs));
  for (int i = 0; i < numOutputs; i++) {
    out.Set(static_cast<uint32_t>(i), MakePort(env, name, i));
  }
  return out;
}

// graphAddGradients(graph, ys, xs) -> Port[]
//
// Adds d(sum(ys))/dx nodes for each x. Coverage comes from TensorFlow's C++
// gradient registry (140 ops in 2.10), so an op without a registered gradient
// fails here — the Status message names it.
Napi::Value GraphAddGradients(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  GraphBox* box = UnwrapGraph(info[0]);
  Napi::Array ysArr = info[1].As<Napi::Array>();
  Napi::Array xsArr = info[2].As<Napi::Array>();

  std::vector<TF_Output> ys(ysArr.Length());
  for (uint32_t i = 0; i < ysArr.Length(); i++) {
    if (!PortFrom(env, box->graph, ysArr.Get(i), &ys[i])) return env.Undefined();
  }
  std::vector<TF_Output> xs(xsArr.Length());
  for (uint32_t i = 0; i < xsArr.Length(); i++) {
    if (!PortFrom(env, box->graph, xsArr.Get(i), &xs[i])) return env.Undefined();
  }

  std::vector<TF_Output> dy(xs.size());
  TF_Status* status = TF_NewStatus();
  TF_AddGradients(box->graph, ys.data(), static_cast<int>(ys.size()), xs.data(),
                  static_cast<int>(xs.size()), nullptr, status, dy.data());
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }
  TF_DeleteStatus(status);

  Napi::Array out = Napi::Array::New(env, dy.size());
  for (size_t i = 0; i < dy.size(); i++) {
    // A gradient can legitimately be absent (unreachable input); report null.
    if (dy[i].oper == nullptr) {
      out.Set(static_cast<uint32_t>(i), env.Null());
      continue;
    }
    out.Set(static_cast<uint32_t>(i),
            MakePort(env, TF_OperationName(dy[i].oper), dy[i].index));
  }
  return out;
}

// graphNewSession(graph) -> SessionHandle (the session takes over the graph)
Napi::Value GraphNewSession(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  GraphBox* box = UnwrapGraph(info[0]);

  TF_Status* status = TF_NewStatus();
  TF_SessionOptions* opts = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(box->graph, opts, status);
  TF_DeleteSessionOptions(opts);
  if (ThrowIfError(env, status)) {
    TF_DeleteStatus(status);
    return env.Undefined();
  }
  TF_DeleteStatus(status);

  TF_Graph* graph = box->graph;
  box->owned = false;  // ownership moves to the session
  return WrapSession(env, session, graph);
}

}  // namespace

void RegisterGraph(Napi::Env env, Napi::Object exports) {
  exports.Set("graphCreate", Napi::Function::New(env, GraphCreate));
  exports.Set("graphPlaceholder", Napi::Function::New(env, GraphPlaceholder));
  exports.Set("graphConst", Napi::Function::New(env, GraphConst));
  exports.Set("graphAddOp", Napi::Function::New(env, GraphAddOp));
  exports.Set("graphAddGradients", Napi::Function::New(env, GraphAddGradients));
  exports.Set("graphNewSession", Napi::Function::New(env, GraphNewSession));
}

}  // namespace tfjs_native
