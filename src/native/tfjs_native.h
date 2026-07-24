// Shared declarations for the tfjs-native N-API addon.
#pragma once

#include <napi.h>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"

namespace tfjs_native {

// Owns a TFE_TensorHandle so it can be freed either eagerly (deleteHandle) or
// lazily (GC finalizer), but never twice. Exposed to JS as a Napi::External.
struct HandleBox {
  TFE_TensorHandle* handle;
};

// Wraps a native handle in a JS External with a GC finalizer.
Napi::External<HandleBox> WrapHandle(Napi::Env env, TFE_TensorHandle* handle);

// Unwraps the HandleBox behind an External value (created by WrapHandle).
HandleBox* UnwrapBox(const Napi::Value& value);

// Lazily-created process-wide eager context (CPU). Returns nullptr and throws a
// JS error on failure.
TFE_Context* GetContext(Napi::Env env);

// A session plus the graph it runs. Used by both SavedModel loading and graphs
// built here for gradients, so they share one run/dispose implementation.
struct SessionBox {
  TF_Session* session;
  TF_Graph* graph;
};

Napi::External<SessionBox> WrapSession(Napi::Env env, TF_Session* session, TF_Graph* graph);
SessionBox* UnwrapSession(const Napi::Value& value);

// Closes the session and frees the graph. Safe to call more than once.
void FreeSession(SessionBox* box);

// Resolves "opName" + output index into a TF_Output, throwing when absent.
// Returns false if a JS error was thrown.
bool ResolveOutput(Napi::Env env, TF_Graph* graph, const std::string& opName, int index,
                   TF_Output* out);

// Registers tensor <-> handle conversion functions on `exports`.
void RegisterTensor(Napi::Env env, Napi::Object exports);

// Registers eager execute / gradient / op-registry functions on `exports`.
void RegisterExecute(Napi::Env env, Napi::Object exports);

// Registers SavedModel loading on `exports`.
void RegisterModel(Napi::Env env, Napi::Object exports);

// Registers session run / dispose on `exports`.
void RegisterSession(Napi::Env env, Napi::Object exports);

// Registers graph construction (and gradients) on `exports`.
void RegisterGraph(Napi::Env env, Napi::Object exports);

// Throws a JS error carrying the TF_Status code+message when `status` is not OK.
// Returns true when an error was thrown (caller should bail out).
bool ThrowIfError(Napi::Env env, TF_Status* status);

}  // namespace tfjs_native
