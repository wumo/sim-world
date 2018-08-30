package wumo.sim.tensorflow.tensor

import wumo.sim.tensorflow.ops.Output

/**
 * Returns the constant value of the given tensor, if efficiently calculable.

This function attempts to partially evaluate the given tensor, and
returns its value as a numpy ndarray if this succeeds.

TODO(mrry): Consider whether this function should use a registration
mechanism like gradients and ShapeFunctions, so that it is easily
extensible.

NOTE: If `constant_value(tensor)` returns a non-`None` result, it will no
longer be possible to feed a different value for `tensor`. This allows the
result of this function to influence the graph that is constructed, and
permits static shape optimizations.

Args:
tensor: The Tensor to be evaluated.
partial: If True, the returned numpy array is allowed to have partially
evaluated values. Values that can't be evaluated will be None.

Returns:
A numpy ndarray containing the constant value of the given `tensor`,
or None if it cannot be calculated.
 
 * @see "tensorflow.python.framework.tensor_util.constant_value"
 */
fun constantValue(tensor: Output): Tensor<*>? {
  when (tensor.op.opType) {
    "Const" -> tensor.op.attr
    "Shape" -> {
      val inputShape = tensor.op.inputs[0].shape
      if (inputShape.isFullyDefined)
        inputShape
    }
  }
  TODO()
}