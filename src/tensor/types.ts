// Shared value types for the tensor layer.

/** A scalar leaf read back from a tensor. `string` tensors decode as UTF-8. */
export type TensorScalar = number | boolean | bigint | string;

/** A scalar or an arbitrarily nested (rectangular) list of scalars. */
export type NestedList = TensorScalar | readonly NestedList[];

/** A scalar leaf accepted on input: raw bytes are allowed for `string` elements. */
export type TensorInputScalar = TensorScalar | Uint8Array;

/** A nested (rectangular) list of input leaves. */
export type NestedInput = TensorInputScalar | readonly NestedInput[];

/** Anything accepted as tensor input: nested lists or a flat TypedArray. */
export type TensorValues = NestedInput | ArrayBufferView;
