{
  "variables": {
    # Root of the extracted libtensorflow C library (headers + lib).
    # Populated by scripts/install.mjs; override with LIBTENSORFLOW_ROOT.
    "libtf_root%": "<!(node -p \"process.env.LIBTENSORFLOW_ROOT || require('path').resolve('deps/libtensorflow')\")"
  },
  "targets": [
    {
      "target_name": "tfjs_native",
      "sources": [
        "src/native/binding.cc",
        "src/native/tensor.cc",
        "src/native/execute.cc"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "<(libtf_root)/include"
      ],
      "library_dirs": [
        "<(libtf_root)/lib"
      ],
      "defines": ["NAPI_VERSION=8", "NODE_ADDON_API_DISABLE_DEPRECATED"],
      "cflags_cc": ["-std=c++17", "-fexceptions"],
      "conditions": [
        ["OS=='linux'", {
          "libraries": ["-ltensorflow", "-Wl,-rpath,'$$ORIGIN/../../deps/libtensorflow/lib'"]
        }],
        ["OS=='mac'", {
          "libraries": ["-ltensorflow", "-Wl,-rpath,@loader_path/../../deps/libtensorflow/lib"],
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
            "MACOSX_DEPLOYMENT_TARGET": "11.0"
          }
        }],
        ["OS=='win'", {
          "libraries": ["tensorflow.lib"],
          "msvs_settings": {
            "VCCLCompilerTool": { "ExceptionHandling": 1, "AdditionalOptions": ["/std:c++17"] }
          }
        }]
      ]
    }
  ]
}
