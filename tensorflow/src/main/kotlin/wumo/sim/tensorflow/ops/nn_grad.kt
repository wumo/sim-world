package wumo.sim.tensorflow.ops

import wumo.sim.tensorflow.ops.gradients.gradient_ops.Registry.register
import wumo.sim.tensorflow.tf
import wumo.sim.tensorflow.types.INT64
import wumo.sim.util.i

fun register_nn_grad() {
  register("Softmax") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
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
    val dyy = tf._mul(grad, y)
    val sum = tf._reshape(tf._sum(dyy, tf.const(i(1))), tf.const(i(-1, 1)))
    val sub = tf._sub(grad, sum)
    val dx = tf._mul(sub, y)
    grad_outputs.add(dx)
  }
  
  register("LogSoftmax") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val softmax = tf._exp(op.outputs[0])
    val sum = tf._sum(grad, tf.const(i(1)), keep_dims = true)
    val mul = tf._mul(sum, softmax)
    val dx = tf._sub(grad, mul)
    grad_outputs.add(dx)
  }
  register("Relu") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val dx = tf._reluGrad(grad, op.inputs[0])
    grad_outputs.add(dx)
  }
  register("Relu6") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val dx = tf._relu6Grad(grad, op.inputs[0])
    grad_outputs.add(dx)
  }
  register("Elu") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val dx = tf._eluGrad(grad, op.outputs[0])
    grad_outputs.add(dx)
  }
  register("Selu") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val dx = tf._seluGrad(grad, op.outputs[0])
    grad_outputs.add(dx)
  }
  register("L2Loss") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    grad_outputs.add(tf._mul(op.inputs[0], grad))
  }
  register("BiasAdd") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val data_format = op.outputs[0].op!!.attrString("data_format")
    val dx_1 = tf._biasAddGrad(grad, data_format = data_format)
    grad_outputs.add(tf._identity(grad))
    grad_outputs.add(dx_1)
  }
  register("Conv2D") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val data_format = op.attrString("data_format")
    val padding = op.attrString("padding")
    val strides = op.attrLongList("strides")
    val use_cudnn_on_gpu = op.attrBool("use_cudnn_on_gpu")
    val dx_1 = tf._conv2DBackpropInput(tf._shape(op.inputs[0]), op.inputs[1], grad, strides.toTypedArray(), padding,
                                       data_format = data_format,
                                       use_cudnn_on_gpu = use_cudnn_on_gpu)
    grad_outputs.add(dx_1)
    val dx_2 = tf._conv2DBackpropFilter(op.inputs[0], tf._shape(op.inputs[1]),
                                        grad, strides.toTypedArray(), padding,
                                        data_format = data_format,
                                        use_cudnn_on_gpu = use_cudnn_on_gpu)
    grad_outputs.add(dx_2)
  }
  register("MaxPool") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val data_format = op.attrString("data_format")
    val padding = op.attrString("padding")
    val strides = op.attrLongList("strides")
    val ksize = op.attrLongList("ksize")
    val dx = tf._maxPoolGrad(
        op.inputs[0], op.outputs[0], grad, ksize.toTypedArray(), strides.toTypedArray(), padding,
        data_format = data_format)
    grad_outputs.add(dx)
  }
  register("MaxPoolV2") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val data_format = op.attrString("data_format")
    val padding = op.attrString("padding")
    val dx = tf._maxPoolGradV2(op.inputs[0], op.outputs[0], grad,
                               op.inputs[1], op.inputs[2], padding, data_format = data_format)
    grad_outputs.add(dx)
    grad_outputs.add(noGradient)
    grad_outputs.add(noGradient)
  }
  register("MaxPool3D") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val data_format = op.attrString("data_format")
    val padding = op.attrString("padding")
    val strides = op.attrLongList("strides")
    val ksize = op.attrLongList("ksize")
    val dx = tf._maxPool3DGrad(op.inputs[0], op.outputs[0], grad,
                               ksize.toTypedArray(), strides.toTypedArray(),
                               padding, data_format = data_format)
    grad_outputs.add(dx)
  }
  register("AvgPool") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val data_format = op.attrString("data_format")
    val padding = op.attrString("padding")
    val strides = op.attrLongList("strides")
    val ksize = op.attrLongList("ksize")
    val dx = tf._avgPoolGrad(tf._shape(op.inputs[0]), grad,
                             ksize.toTypedArray(), strides.toTypedArray(), padding,
                             data_format = data_format)
    grad_outputs.add(dx)
  }
  register("AvgPool3D") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val data_format = op.attrString("data_format")
    val padding = op.attrString("padding")
    val strides = op.attrLongList("strides")
    val ksize = op.attrLongList("ksize")
    val dx = tf._avgPool3DGrad(tf._shape(op.inputs[0]), grad,
                               ksize.toTypedArray(), strides.toTypedArray(), padding,
                               data_format = data_format)
    grad_outputs.add(dx)
  }
  register("LRN") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val dx = tf._lRNGrad(grad, op.inputs[0], op.outputs[0])
    grad_outputs.add(dx)
  }
  register("Softplus") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val dx = tf._softplusGrad(grad, op.inputs[0])
    grad_outputs.add(dx)
  }
  register("Softsign") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val dx = tf._softsignGrad(grad, op.inputs[0]);
    grad_outputs.add(dx)
  }
  register("FractionalAvgPool") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val overlapping = op.attrBool("overlapping")
    val dx = tf._fractionalAvgPoolGrad(
        tf._shape(op.inputs[0], out_type = INT64),
        grad, op.outputs[1], op.outputs[2],
        overlapping = overlapping)
    grad_outputs.add(dx)
  }
  register("FractionalMaxPool") { op, grad_inputs, grad_outputs ->
    val grad = grad_inputs[0]!!.toOutput()
    val overlapping = op.attrBool("overlapping")
    val dx = tf._fractionalMaxPoolGrad(
        op.inputs[0], op.outputs[0], grad, op.outputs[1], op.outputs[2],
        overlapping = overlapping)
    grad_outputs.add(dx)
  }
}