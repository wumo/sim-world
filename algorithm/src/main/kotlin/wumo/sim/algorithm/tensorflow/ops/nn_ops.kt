package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import org.bytedeco.javacpp.tensorflow.DT_HALF
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.util.a
import wumo.sim.util.i

fun TF.avgPool3DGrad(orig_input_shape: Tensor,
                     grad: Tensor,
                     ksize: LongArray,
                     strides: LongArray,
                     padding: String,
                     data_format: String = "NHWC",
                     name: String = "AvgPool3DGrad") =
    naryOp("AvgPool3DGrad", orig_input_shape.value(), grad.value(), name = name) {
      attr("ksize", ksize)
      attr("strides", strides)
      attr("padding", padding)
      attr("data_format", data_format)
    }

fun TF.biasAdd(value: Tensor, bias: Tensor, name: String = "BiasAdd") =
    naryOp("BiasAdd", value.value(), bias.value(), name = name) {
      attr("data_format", "NHWC")
    }

fun TF.conv2D(input: Tensor,
              filter: Tensor,
              strides: LongArray,
              padding: String,
              use_cudnn_on_gpu: Boolean = true,
              data_format: String = "NHWC",
              dilations: IntArray = i(1, 1, 1, 1),
              name: String = "Conv2D") =
    naryOp("Conv2D", input.value(), filter.value(), name = name) {
      attr("strides", strides)
      attr("use_cudnn_on_gpu", use_cudnn_on_gpu)
      attr("padding", padding)
      attr("data_format", data_format)
      attr("dilations", dilations)
    }

fun TF.conv2DBackpropFilter(input: Tensor,
                            filter_sizes: Tensor,
                            out_backprop: Tensor,
                            strides: LongArray,
                            padding: String,
                            use_cudnn_on_gpu: Boolean = true,
                            data_format: String = "NHWC",
                            dilations: IntArray = i(1, 1, 1, 1),
                            name: String = "Conv2DBackpropFilter") =
    naryOp("Conv2DBackpropFilter", input.value(), filter_sizes.value(), out_backprop.value(), name = name) {
      attr("strides", strides)
      attr("use_cudnn_on_gpu", use_cudnn_on_gpu)
      attr("padding", padding)
      attr("data_format", data_format)
      attr("dilations", dilations)
    }

fun TF.conv2DBackpropInput(input_sizes: Tensor,
                           filter: Tensor,
                           out_backprop: Tensor,
                           strides: LongArray,
                           padding: String,
                           use_cudnn_on_gpu: Boolean = true,
                           data_format: String = "NHWC",
                           dilations: IntArray = i(1, 1, 1, 1),
                           name: String = "Conv2DBackpropInput") =
    naryOp("Conv2DBackpropInput", input_sizes.value(), filter.value(), out_backprop.value(), name = name) {
      attr("strides", strides)
      attr("use_cudnn_on_gpu", use_cudnn_on_gpu)
      attr("padding", padding)
      attr("data_format", data_format)
      attr("dilations", dilations)
    }

fun TF.maxPool3DGrad(orig_input: Tensor,
                     orig_output: Tensor,
                     grad: Tensor,
                     ksize: LongArray,
                     strides: LongArray,
                     padding: String,
                     data_format: String = "NHWC",
                     name: String = "MaxPool3DGrad") =
    naryOp("MaxPool3DGrad", orig_input.value(), orig_output.value(), grad.value(), name = name) {
      attr("ksize", ksize)
      attr("strides", strides)
      attr("padding", padding)
      attr("data_format", data_format)
    }

fun TF.maxPool3DGradGrad(orig_input: Tensor,
                         orig_output: Tensor,
                         grad: Tensor,
                         ksize: LongArray,
                         strides: LongArray,
                         padding: String,
                         data_format: String = "NHWC",
                         name: String = "MaxPool3DGradGrad") =
    naryOp("MaxPool3DGradGrad", orig_input.value(), orig_output.value(), grad.value(), name = name) {
      attr("ksize", ksize)
      attr("strides", strides)
      attr("padding", padding)
      attr("data_format", data_format)
    }

fun TF.maxPoolGradGrad(orig_input: Tensor,
                       orig_output: Tensor,
                       grad: Tensor,
                       ksize: LongArray,
                       strides: LongArray,
                       padding: String,
                       data_format: String = "NHWC",
                       name: String = "MaxPoolGradGrad") =
    naryOp("MaxPoolGradGrad", orig_input.value(), orig_output.value(), grad.value(), name = name) {
      attr("ksize", ksize)
      attr("strides", strides)
      attr("padding", padding)
      attr("data_format", data_format)
    }

fun TF.maxPoolGradGradV2(orig_input: Tensor,
                         orig_output: Tensor,
                         grad: Tensor,
                         ksize: Tensor,
                         strides: Tensor,
                         padding: String,
                         data_format: String = "NHWC",
                         name: String = "MaxPoolGradGradV2") =
    naryOp("MaxPoolGradGradV2", orig_input.value(), orig_output.value(),
           grad.value(), ksize.value(), strides.value(), name = name) {
      attr("padding", padding)
      attr("data_format", data_format)
    }

fun TF.maxPoolGradV2(orig_input: Tensor,
                     orig_output: Tensor,
                     grad: Tensor,
                     ksize: Tensor,
                     strides: Tensor,
                     padding: String,
                     data_format: String = "NHWC",
                     name: String = "MaxPoolGradV2") =
    naryOp("MaxPoolGradV2", orig_input.value(), orig_output.value(),
           grad.value(), ksize.value(), strides.value(), name = name) {
      attr("padding", padding)
      attr("data_format", data_format)
    }

fun TF.relu(features: Tensor, name: String = "Relu") =
    unaryOp("Relu", features.value(), name)

fun TF.softmax(logits: Tensor, name: String = "Softmax") =
    unaryOp("Softmax", logits.value(), name)

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
 
 * @param x: A `Tensor`.
 * @param axes: Array of ints.  Axes along which to compute mean and
variance.
 * @param name: Name used to scope the operations that compute the moments.
 * @param keep_dims: produce moments with the same dimensionality as the input.
 
 * @return: Two `Tensor` objects: `mean` and `variance`.
 */
fun TF.moments(x: Tensor,
               axes: IntArray,
               name: String = "moments",
               keep_dims: Boolean = false): Array<Tensor> {
  name_scope(name) {
    //The dynamic range of fp16 is too limited to support the collection of
    //sufficient statistics. As a workaround we simply perform the operations
    //on 32-bit floats before converting the mean and variance back to fp16
    val y = if (x.dtype == DT_HALF) tf.cast(x, DT_FLOAT) else x
    //Compute true mean while keeping the dims for proper broadcasting.
    var mean = tf.mean(y, axes, keep_dims = true, name = "mean")
    //sample variance, not unbiased variance
    var variance = tf.mean(tf.squaredDifference(y, tf.stop_gradient(mean)),
                           axes, keep_dims = true, name = "variance")
    if (!keep_dims) {
      mean = tf.squeeze(mean, axes)
      variance = tf.squeeze(variance, axes)
    }
    return if (x.dtype == DT_HALF)
      a(tf.cast(mean, DT_HALF), tf.cast(variance, DT_HALF))
    else a(mean, variance)
  }
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
 
 * @param x: Input `Tensor` of arbitrary dimensionality.
 * @param mean: A mean `Tensor`.
 * @param variance: A variance `Tensor`.
 * @param offset: An offset `Tensor`, often denoted \\(\beta\\) in equations, or
None. If present, will be added to the normalized tensor.
 * @param scale: A scale `Tensor`, often denoted \\(\gamma\\) in equations, or
`None`. If present, the scale is applied to the normalized tensor.
 * @param variance_epsilon: A small float number to avoid dividing by 0.
 * @param name: A name for this operation (optional).
 
 * @return:the normalized, scaled, offset tensor.
 */
fun TF.batch_normalization(x: Tensor,
                           mean: Tensor, variance: Tensor,
                           offset: Tensor?, scale: Tensor?,
                           variance_epsilon: Float,
                           name: String = "batchnorm"): Tensor {
  name_scope(name) {
    //    val _variance_epsilon = tf.const(variance_epsilon)
    var inv = tf.rsqrt(variance + variance_epsilon)
    if (scale != null)
      inv *= scale
    return x * inv + (if (offset != null) offset - mean * inv else -mean * inv)
  }
}