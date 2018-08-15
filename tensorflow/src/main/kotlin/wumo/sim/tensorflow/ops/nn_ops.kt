package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.FLOAT
import wumo.sim.tensorflow.types.HALF
import wumo.sim.util.a

object nn_ops {
  interface API {
    
    /**
     * Calculate the mean and variance of `x`.
    
    The mean and variance are calculated by aggregating the contents of `x`
    across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
    and variance of a vector.
    
    Note: shift is currently not used; the true mean is computed and used.
    
    When using these moments for batch normalization (see
    `tf.nn.batch_normalization`):
     
     * for so-called "global normalization", used with convolutional filters with
    shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
     * for simple batch normalization pass `axes=[0]` (batch only).
     
     * @param x: A `Output`.
     * @param axes: Array of ints.  Axes along which to compute mean and
    variance.
     * @param name: Name used to scope the operations that compute the moments.
     * @param keep_dims: produce moments with the same dimensionality as the input.
     
     * @return: Two `Output` objects: `mean` and `variance`.
     */
    fun moments(x: Output,
                axes: LongArray,
                name: String = "moments",
                keep_dims: Boolean = false): Array<Output> =
        tf.name_scope(name) {
          //The dynamic range of fp16 is too limited to support the collection of
          //sufficient statistics. As a workaround we simply perform the operations
          //on 32-bit floats before converting the mean and variance back to fp16
          val y = if (x.dtype == HALF) tf.cast(x, FLOAT) else x
          //Compute true mean while keeping the dims for proper broadcasting.
          var mean = tf.mean(y, axes, keep_dims = true, name = "mean")
          //sample variance, not unbiased variance
          var variance = tf.mean(tf._squaredDifference(y, tf._stopGradient(mean)),
                                 axes, keep_dims = true, name = "variance")
          if (!keep_dims) {
            mean = tf._squeeze(mean, axes.toTypedArray())
            variance = tf._squeeze(variance, axes.toTypedArray())
          }
          if (x.dtype == HALF)
            a(tf.cast(mean, HALF), tf.cast(variance, HALF))
          else a(mean, variance)
        }
    
    /**
     * Batch normalization.
     * As described in http://arxiv.org/abs/1502.03167.
    Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
    `scale` \\(\gamma\\) to it, as well as an `offset` \\(\beta\\):
    
    \\(\frac{\gamma(x-\mu)}{\sigma}+\beta\\)
    
    `mean`, `variance`, `offset` and `scale` are all expected to be of one of two
    shapes:
     
     * In all generality, they can have the same number of dimensions as the
    input `x`, with identical sizes as `x` for the dimensions that are not
    normalized over (the 'depth' dimension(s)), and dimension 1 for the
    others which are being normalized over.
    `mean` and `variance` in this case would typically be the outputs of
    `tf.nn.moments(..., keep_dims=True)` during training, or running averages
    thereof during inference.
     * In the common case where the 'depth' dimension is the last dimension in
    the input tensor `x`, they may be one dimensional tensors of the same
    size as the 'depth' dimension.
    This is the case for example for the common `[batch, depth]` layout of
    fully-connected layers, and `[batch, height, width, depth]` for
    convolutions.
    `mean` and `variance` in this case would typically be the outputs of
    `tf.nn.moments(..., keep_dims=False)` during training, or running averages
    thereof during inference.
     
     * @param x: Input `Output` of arbitrary dimensionality.
     * @param mean: A mean `Output`.
     * @param variance: A variance `Output`.
     * @param offset: An offset `Output`, often denoted \\(\beta\\) in equations, or
    None. If present, will be added to the normalized tensor.
     * @param scale: A scale `Output`, often denoted \\(\gamma\\) in equations, or
    `None`. If present, the scale is applied to the normalized tensor.
     * @param variance_epsilon: A small float number to avoid dividing by 0.
     * @param name: A name for this operation (optional).
     
     * @return:the normalized, scaled, offset tensor.
     */
    fun batch_normalization(x: Output,
                            mean: Output, variance: Output,
                            offset: Output?, scale: Output?,
                            variance_epsilon: Float,
                            name: String = "batchnorm"): Output =
        tf.name_scope(name) {
          //    val _variance_epsilon = tf.const(variance_epsilon)
          var inv = tf._rsqrt(variance + variance_epsilon)
          if (scale != null)
            inv *= scale
          x * inv + (if (offset != null) offset - mean * inv else -mean * inv)
        }
  }
}