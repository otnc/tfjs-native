// Single source of truth for TF <-> JS dtype mapping (see docs/DESIGN.md §4).
// The numeric values match TF_DataType in tensorflow/c/tf_datatype.h.

export const DType = {
  float32: 1,
  float64: 2,
  int32: 3,
  uint8: 4,
  int64: 9,
  bool: 10,
  complex64: 8,
  string: 7,
} as const;

export type DTypeName = keyof typeof DType;
export type DTypeValue = (typeof DType)[DTypeName];

/** Default dtype for tensors created from plain JS numbers. */
export const DEFAULT_DTYPE: DTypeName = "float32";

const NAME_BY_VALUE = new Map<number, DTypeName>(
  (Object.entries(DType) as [DTypeName, number][]).map(([name, value]) => [value, name]),
);

/** Maps a TF_DataType numeric code back to its dtype name. */
export function dtypeName(value: number): DTypeName {
  const name = NAME_BY_VALUE.get(value);
  if (name === undefined) {
    throw new Error(`tfjs-native: unknown TF_DataType value ${value}`);
  }
  return name;
}

/** TypedArray constructor used to read/write each dtype's raw buffer. */
export const TypedArrayFor = {
  float32: Float32Array,
  float64: Float64Array,
  int32: Int32Array,
  uint8: Uint8Array,
  int64: BigInt64Array,
  bool: Uint8Array,
  complex64: Float32Array,
} as const;
