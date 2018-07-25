package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.binaryOp
import wumo.sim.algorithm.tensorflow.naryOp

fun TF.avgPoolGrad(orig_input_shape: Tensor,
                   grad: Tensor,
                   ksize: IntArray, strides: IntArray,
                   padding: String,
                   data_format: String = "NHWC",
                   name: String = "AvgPoolGrad") =
    naryOp("AvgPoolGrad", orig_input_shape, grad, name = name) {
      attr("ksize", ksize)
      attr("strides", strides)
      attr("padding", padding)
      attr("data_format", data_format)
    }

fun TF.eluGrad(gradients: Tensor,
               outputs: Tensor,
               name: String = "EluGrad") =
    binaryOp("EluGrad", gradients, outputs, name = name)

fun TF.fractionalAvgPoolGrad(orig_input_tensor_shape: Tensor,
                             out_backprop: Tensor,
                             row_polling_sequence: Tensor,
                             col_pooling_sequence: Tensor,
                             overlapping: Boolean = false,
                             name: String = "FractionalAvgPoolGrad") =
    naryOp("FractionalAvgPoolGrad", orig_input_tensor_shape, out_backprop, row_polling_sequence, col_pooling_sequence, name = name) {
      attr("overlapping", overlapping)
    }

fun TF.fractionalMaxPoolGrad(orig_input: Tensor,
                             orig_output: Tensor,
                             out_backprop: Tensor,
                             row_pooling_sequence: Tensor,
                             col_pooling_sequence: Tensor,
                             overlapping: Boolean = false,
                             name: String = "FractionalMaxPoolGrad") =
    naryOp("FractionalMaxPoolGrad", orig_input, orig_output, out_backprop, row_pooling_sequence, col_pooling_sequence, name = name) {
      attr("overlapping", overlapping)
    }

fun TF.LRNGrad(intput_grads: Tensor,
               input_image: Tensor,
               output_image: Tensor,
               depth_radius: Long = 5L,
               bias: Float = 1f,
               alpha: Float = 1f,
               beta: Float = 0.5f,
               name: String = "LRNGrad") =
    naryOp("LRNGrad", intput_grads, input_image, output_image, name = name) {
      attr("depth_radius", depth_radius)
      attr("bias", bias)
      attr("alpha", alpha)
      attr("beta", beta)
    }

fun TF.maxPoolGrad(orig_input: Tensor,
                   orig_output: Tensor,
                   grad: Tensor,
                   ksize: IntArray, strides: IntArray,
                   padding: String,
                   data_format: String = "NHWC",
                   name: String = "MaxPoolGrad") =
    naryOp("MaxPoolGrad", orig_input, orig_output, grad, name = name) {
      attr("ksize", ksize)
      attr("strides", strides)
      attr("padding", padding)
      attr("data_format", data_format)
    }

fun TF.maxPoolGradWithArgmax(input: Tensor,
                             grad: Tensor,
                             argmax: Tensor,
                             ksize: IntArray, strides: IntArray,
                             padding: String,
                             name: String = "MaxPoolGradWithArgmax") =
    naryOp("MaxPoolGradWithArgmax", input, grad, argmax, name = name) {
      attr("ksize", ksize)
      attr("strides", strides)
      attr("padding", padding)
    }

fun TF.relu6Grad(gradients: Tensor,
                 features: Tensor,
                 name: String = "Relu6Grad") =
    binaryOp("Relu6Grad", gradients, features, name = name)

fun TF.reluGrad(gradients: Tensor,
                features: Tensor,
                name: String = "ReluGrad") =
    binaryOp("ReluGrad", gradients, features, name = name)

fun TF.seluGrad(gradients: Tensor,
                features: Tensor,
                name: String = "SeluGrad") =
    binaryOp("SeluGrad", gradients, features, name = name)

fun TF.softplusGrad(gradients: Tensor,
                    features: Tensor,
                    name: String = "SoftplusGrad") =
    binaryOp("SoftplusGrad", gradients, features, name = name)

fun TF.softsignGrad(gradients: Tensor,
                    features: Tensor,
                    name: String = "SoftsignGrad") =
    binaryOp("SoftsignGrad", gradients, features, name = name)