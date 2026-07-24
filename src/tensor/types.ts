// Shared value types for the tensor layer.

/** A scalar leaf accepted when building a tensor from plain JS. */
export type TensorScalar = number | boolean | bigint;

/** A scalar or an arbitrarily nested (rectangular) list of scalars. */
export type NestedList = TensorScalar | readonly NestedList[];

/** Anything accepted as tensor input: nested lists or a flat TypedArray. */
export type TensorValues = NestedList | ArrayBufferView;
