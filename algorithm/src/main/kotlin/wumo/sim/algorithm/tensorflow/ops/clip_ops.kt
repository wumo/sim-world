package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.tf

/**
 * Clips tensor values to a maximum L2-norm.
Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
normalizes `t` so that its L2-norm is less than or equal to `clip_norm`,
along the dimensions given in `axes`. Specifically, in the default case
where all dimensions are used for calculation, if the L2-norm of `t` is
already less than or equal to `clip_norm`, then `t` is not modified. If
the L2-norm is greater than `clip_norm`, then this operation returns a
tensor of the same type and shape as `t` with its values set to:

`t * clip_norm / l2norm(t)`

In this case, the L2-norm of the output tensor is `clip_norm`.

As another example, if `t` is a matrix and `axes == [1]`, then each row
of the output will have L2-norm equal to `clip_norm`. If `axes == [0]`
instead, each column of the output will be clipped.

This operation is typically used to clip gradients before applying them with
an optimizer.
 
 
 * @param t: A `Tensor`.
 * @param clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
 * @param axes: A 1-D (vector) `Tensor` of type int32 containing the dimensions
to use for computing the L2-norm. If `None` (the default), uses all
dimensions.
 * @param name: A name for the operation (optional).
 
 * @return A clipped `Tensor`.
 */
fun TF.clip_by_norm(t: Tensor, clip_norm: Tensor, axes: Tensor? = null, name: String = "clip_by_norm"): Tensor {
  name_scope(name) {
    val clip_norm = tf.cast(clip_norm, t.dtype)
    // Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    val l2norm = tf.sqrt(tf.sum(t * t, axes, keep_dims = true))
    val intermediate = t * clip_norm
    // Assert that the shape is compatible with the initial shape,
    // to prevent unintentional broadcasting.
//    t.shape x intermediate.shape
    assert(t.shape.isCompatibleWith(intermediate.shape))
    return tf.identity(
        intermediate / tf.maximum(l2norm, clip_norm), name = ctxNs.scopeNameForOnce())
  }
}