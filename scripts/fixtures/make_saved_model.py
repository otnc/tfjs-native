"""Generate the tiny SavedModel test fixture used by the M3 integration tests.

The model is deliberately trivial (y = x * 2) so the fixture stays small and only
uses ops available in the pinned libtensorflow. TensorFlow's Python package is
needed only to *produce* the fixture — tfjs-native itself never depends on it.

Regenerate (Python 3.12; TensorFlow has no 3.13/3.14 wheels):

    uv venv --python 3.12 .venv-tf
    uv pip install --python .venv-tf tensorflow-cpu
    .venv-tf/Scripts/python scripts/fixtures/make_saved_model.py   # Windows
    .venv-tf/bin/python scripts/fixtures/make_saved_model.py       # macOS/Linux
"""

import os

import tensorflow as tf

OUT_DIR = os.path.join("test", "fixtures", "times_two")


class TimesTwo(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32, name="x")])
    def __call__(self, x):
        return {"y": x * 2.0}


def main() -> None:
    model = TimesTwo()
    tf.saved_model.save(
        model,
        OUT_DIR,
        signatures={"serving_default": model.__call__.get_concrete_function()},
    )
    print(f"wrote {OUT_DIR} (tensorflow {tf.__version__})")


if __name__ == "__main__":
    main()
