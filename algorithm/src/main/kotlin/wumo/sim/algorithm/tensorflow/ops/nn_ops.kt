package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.naryOp
import wumo.sim.algorithm.tensorflow.unaryOp
import wumo.sim.util.i

fun TF.avgPool3DGrad(orig_input_shape: Tensor,
                     grad: Tensor,
                     ksize: IntArray,
                     strides: IntArray,
                     padding: String,
                     data_format: String = "NHWC",
                     name: String = "AvgPool3DGrad") =
    naryOp("AvgPool3DGrad", orig_input_shape, grad, name = name) {
      attr("ksize", ksize)
      attr("strides", strides)
      attr("padding", padding)
      attr("data_format", data_format)
    }

fun TF.biasAdd(value: Tensor, bias: Tensor, name: String = "BiasAdd"): Tensor {
  val op = g.nodeBuilder("BiasAdd", ctxNs.getUniqueFullName(name))
      .addInput(value)
      .addInput(bias)
      .attr("data_format", "NHWC")
      .build()
  return Tensor(op, 0)
}

fun TF.conv2D(input: Tensor,
              filter: Tensor,
              strides: IntArray,
              padding: String,
              use_cudnn_on_gpu: Boolean = true,
              data_format: String = "NHWC",
              dilations: IntArray = i(1, 1, 1, 1),
              name: String = "Conv2D") =
    naryOp("Conv2D", input, filter, name = name) {
      attr("strides", strides)
      attr("use_cudnn_on_gpu", use_cudnn_on_gpu)
      attr("padding", padding)
      attr("data_format", data_format)
      attr("dilations", dilations)
    }

fun TF.conv2DBackpropFilter(input: Tensor,
                            filter_sizes: Tensor,
                            out_backprop: Tensor,
                            strides: IntArray,
                            padding: String,
                            use_cudnn_on_gpu: Boolean = true,
                            data_format: String = "NHWC",
                            dilations: IntArray = i(1, 1, 1, 1),
                            name: String = "Conv2DBackpropFilter") =
    naryOp("Conv2DBackpropFilter", input, filter_sizes, out_backprop, name = name) {
      attr("strides", strides)
      attr("use_cudnn_on_gpu", use_cudnn_on_gpu)
      attr("padding", padding)
      attr("data_format", data_format)
      attr("dilations", dilations)
    }

fun TF.conv2DBackpropInput(input_sizes: Tensor,
                           filter: Tensor,
                           out_backprop: Tensor,
                           strides: IntArray,
                           padding: String,
                           use_cudnn_on_gpu: Boolean = true,
                           data_format: String = "NHWC",
                           dilations: IntArray = i(1, 1, 1, 1),
                           name: String = "Conv2DBackpropInput") =
    naryOp("Conv2DBackpropInput", input_sizes, filter, out_backprop, name = name) {
      attr("strides", strides)
      attr("use_cudnn_on_gpu", use_cudnn_on_gpu)
      attr("padding", padding)
      attr("data_format", data_format)
      attr("dilations", dilations)
    }

fun TF.maxPool3DGrad(orig_input: Tensor,
                     orig_output: Tensor,
                     grad: Tensor,
                     ksize: IntArray,
                     strides: IntArray,
                     padding: String,
                     data_format: String = "NHWC",
                     name: String = "MaxPool3DGrad") =
    naryOp("MaxPool3DGrad", orig_input, orig_output, grad, name = name) {
      attr("ksize", ksize)
      attr("strides", strides)
      attr("padding", padding)
      attr("data_format", data_format)
    }

fun TF.maxPool3DGradGrad(orig_input: Tensor,
                         orig_output: Tensor,
                         grad: Tensor,
                         ksize: IntArray,
                         strides: IntArray,
                         padding: String,
                         data_format: String = "NHWC",
                         name: String = "MaxPool3DGradGrad") =
    naryOp("MaxPool3DGradGrad", orig_input, orig_output, grad, name = name) {
      attr("ksize", ksize)
      attr("strides", strides)
      attr("padding", padding)
      attr("data_format", data_format)
    }

fun TF.maxPoolGradGrad(orig_input: Tensor,
                       orig_output: Tensor,
                       grad: Tensor,
                       ksize: IntArray,
                       strides: IntArray,
                       padding: String,
                       data_format: String = "NHWC",
                       name: String = "MaxPoolGradGrad") =
    naryOp("MaxPoolGradGrad", orig_input, orig_output, grad, name = name) {
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
    naryOp("MaxPoolGradGradV2", orig_input, orig_output, grad, ksize, strides, name = name) {
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
    naryOp("MaxPoolGradV2", orig_input, orig_output, grad, ksize, strides, name = name) {
      attr("padding", padding)
      attr("data_format", data_format)
    }

fun TF.relu(features: Tensor, name: String = "Relu") =
    unaryOp("Relu", features, name)

fun TF.softmax(logits: Tensor, name: String = "Softmax") =
    unaryOp("Softmax", logits, name)