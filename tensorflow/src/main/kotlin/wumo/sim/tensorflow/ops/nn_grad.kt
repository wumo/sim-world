package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_INT64
import wumo.sim.algorithm.tensorflow.ops.gen.*
import wumo.sim.algorithm.tensorflow.ops.gradients.noGradient
import wumo.sim.algorithm.tensorflow.ops.gradients.register_gradient_op
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.i

fun register_nn_grad() {
  register_gradient_op("Softmax") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    // Softmax gradient function.
    // p = softmax(x) maps from [batch, n] to [batch, m]
    // dp/dx = [dp0/dx0   ... dp0/dxn-1  ]
    //         [  ...           ...      ]
    //         [dpm-1/dx0 ... dpm-1/dxn-1]
    // dL/dx = dp/dx * dL/dy
    //
    // Using alternative formula:
    // dL/dx = dL/dy * y - sum(dL/dy * y) * y
    //    = (dL/dy - sum(dL/dy * y)) * y
    val y = op.outputs[0]
    val dyy = tf.mul(grad, y)
    val sum = tf.reshape(tf.sum(dyy, tf.const(i(1))), tf.const(i(-1, 1)))
    val sub = tf.sub(grad, sum)
    val dx = tf.mul(sub, y)
    grad_outputs.add(dx)
  }
  
  register_gradient_op("LogSoftmax") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val softmax = tf.exp(op.outputs[0])
    val sum = tf.sum(grad, tf.const(i(1)), keep_dims = true)
    val mul = tf.mul(sum, softmax)
    val dx = tf.sub(grad, mul)
    grad_outputs.add(dx)
  }
  register_gradient_op("Relu") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val dx = tf.reluGrad(grad, op.inputs[0])
    grad_outputs.add(dx)
  }
  register_gradient_op("Relu6") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val dx = tf.relu6Grad(grad, op.inputs[0])
    grad_outputs.add(dx)
  }
  register_gradient_op("Elu") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val dx = tf.eluGrad(grad, op.outputs[0])
    grad_outputs.add(dx)
  }
  register_gradient_op("Selu") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val dx = tf.seluGrad(grad, op.outputs[0])
    grad_outputs.add(dx)
  }
  register_gradient_op("L2Loss") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    grad_outputs.add(tf.mul(op.inputs[0], grad))
  }
  register_gradient_op("BiasAdd") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val data_format = op.outputs[0].op!!.attrString("data_format")
    val dx_1 = tf.biasAddGrad(grad, data_format = data_format)
    grad_outputs.add(tf.identity(grad))
    grad_outputs.add(dx_1)
  }
  register_gradient_op("Conv2D") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val data_format = op.attrString("data_format")
    val padding = op.attrString("padding")
    val strides = op.attrLongList("strides")
    val use_cudnn_on_gpu = op.attrBool("use_cudnn_on_gpu")
    val dx_1 = tf.conv2DBackpropInput(tf.shape(op.inputs[0]), op.inputs[1], grad, strides.toTypedArray(), padding,
                                      data_format = data_format,
                                      use_cudnn_on_gpu = use_cudnn_on_gpu)
    grad_outputs.add(dx_1)
    val dx_2 = tf.conv2DBackpropFilter(op.inputs[0], tf.shape(op.inputs[1]),
                                       grad, strides.toTypedArray(), padding,
                                       data_format = data_format,
                                       use_cudnn_on_gpu = use_cudnn_on_gpu)
    grad_outputs.add(dx_2)
  }
  register_gradient_op("MaxPool") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val data_format = op.attrString("data_format")
    val padding = op.attrString("padding")
    val strides = op.attrLongList("strides")
    val ksize = op.attrLongList("ksize")
    val dx = tf.maxPoolGrad(
        op.inputs[0], op.outputs[0], grad, ksize.toTypedArray(), strides.toTypedArray(), padding,
        data_format = data_format)
    grad_outputs.add(dx)
  }
  register_gradient_op("MaxPoolV2") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val data_format = op.attrString("data_format")
    val padding = op.attrString("padding")
    val dx = tf.maxPoolGradV2(op.inputs[0], op.outputs[0], grad,
                              op.inputs[1], op.inputs[2], padding, data_format = data_format)
    grad_outputs.add(dx)
    grad_outputs.add(noGradient)
    grad_outputs.add(noGradient)
  }
  register_gradient_op("MaxPool3D") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val data_format = op.attrString("data_format")
    val padding = op.attrString("padding")
    val strides = op.attrLongList("strides")
    val ksize = op.attrLongList("ksize")
    val dx = tf.maxPool3DGrad(op.inputs[0], op.outputs[0], grad,
                              ksize.toTypedArray(), strides.toTypedArray(),
                              padding, data_format = data_format)
    grad_outputs.add(dx)
  }
  register_gradient_op("AvgPool") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val data_format = op.attrString("data_format")
    val padding = op.attrString("padding")
    val strides = op.attrLongList("strides")
    val ksize = op.attrLongList("ksize")
    val dx = tf.avgPoolGrad(tf.shape(op.inputs[0]), grad,
                            ksize.toTypedArray(), strides.toTypedArray(), padding,
                            data_format = data_format)
    grad_outputs.add(dx)
  }
  register_gradient_op("AvgPool3D") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val data_format = op.attrString("data_format")
    val padding = op.attrString("padding")
    val strides = op.attrLongList("strides")
    val ksize = op.attrLongList("ksize")
    val dx = tf.avgPool3DGrad(tf.shape(op.inputs[0]), grad,
                              ksize.toTypedArray(), strides.toTypedArray(), padding,
                              data_format = data_format)
    grad_outputs.add(dx)
  }
  register_gradient_op("LRN") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val dx = tf.lRNGrad(grad, op.inputs[0], op.outputs[0])
    grad_outputs.add(dx)
  }
  register_gradient_op("Softplus") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val dx = tf.softplusGrad(grad, op.inputs[0])
    grad_outputs.add(dx)
  }
  register_gradient_op("Softsign") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val dx = tf.softsignGrad(grad, op.inputs[0]);
    grad_outputs.add(dx)
  }
  register_gradient_op("FractionalAvgPool") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val overlapping = op.attrBool("overlapping")
    val dx = tf.fractionalAvgPoolGrad(
        tf.shape(op.inputs[0], out_type = DT_INT64),
        grad, op.outputs[1], op.outputs[2],
        overlapping = overlapping)
    grad_outputs.add(dx)
  }
  register_gradient_op("FractionalMaxPool") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]
    val overlapping = op.attrBool("overlapping")
    val dx = tf.fractionalMaxPoolGrad(
        op.inputs[0], op.outputs[0], grad, op.outputs[1], op.outputs[2],
        overlapping = overlapping)
    grad_outputs.add(dx)
  }
}