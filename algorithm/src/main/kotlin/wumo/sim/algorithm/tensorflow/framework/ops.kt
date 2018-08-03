package wumo.sim.algorithm.tensorflow.framework

import wumo.sim.algorithm.tensorflow.Tensor

/**
Converts the given `value` to a `Tensor`.

This function converts Python objects of various types to `Tensor`
objects. It accepts `Tensor` objects, numpy arrays, Python lists,
and Python scalars. For example:

```python
import numpy as np

def my_func(arg):
arg = tf.convert_to_tensor(arg, dtype=tf.float32)
return tf.matmul(arg, arg) + arg

# The following calls are equivalent.
value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
```

This function can be useful when composing a new findOp in Python
(such as `my_func` in the example above). All standard Python op
constructors apply this function to each of their Tensor-valued
inputs, which allows those ops to accept numpy arrays, Python lists,
and scalars in addition to `Tensor` objects.

Note: This function diverges from default Numpy behavior for `float` and
`string` types when `None` is present in a Python list or scalar. Rather
than silently converting `None` values, an error will be thrown.

Args:
value: An object whose type has a registered `Tensor` conversion function.
dtype: Optional element type for the returned tensor. If missing, the
type is inferred from the type of `value`.
name: Optional name to use if a new `Tensor` is created.
preferred_dtype: Optional element type for the returned tensor,
used when dtype is None. In some cases, a caller may not have a
dtype in mind when converting to a tensor, so preferred_dtype
can be used as a soft preference.  If the conversion to
`preferred_dtype` is not possible, this argument has no effect.

Returns:
An `Output` based on `value`.
 */
fun convert_to_tensor(value: Any, dtype: Int? = null,
                      name: String? = null, preferred_dtype: Int? = null) =
    internal_convert_to_tensor(
        value = value,
        dtype = dtype,
        name = name,
        preferred_dtype = preferred_dtype,
        as_ref = false)

private val tensor_conversion_func_cache = hashMapOf<Class<*>, (Any) -> Tensor>()
private fun internal_convert_to_tensor(
    value: Any,
    dtype: Int? = null,
    name: String? = null,
    as_ref: Boolean = false,
    preferred_dtype: Int? = null): Tensor {
  TODO()
}