package wumo.sim.algorithm.tensorflow.ops

import wumo.sim.algorithm.tensorflow.ops.gradients.register_gradient_op
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.i

fun register_nn_grad() {
  register_gradient_op("Softmax") { op, grad_inputs, grad_outputs ->
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
    val dyy = tf.mul(grad_inputs[0], y)
    val sum = tf.reshape(tf.sum(dyy, tf.const(i(1))), tf.const(i(-1, 1)))
    val sub = tf.sub(grad_inputs[0], sum)
    val dx = tf.mul(sub, y)
    grad_outputs.add(dx)
  }
  
  register_gradient_op("LogSoftmax") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("Relu") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("Relu6") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("Elu") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("Selu") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("L2Loss") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("BiasAdd") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("Conv2D") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("MaxPool") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("MaxPoolV2") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("MaxPool3D") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("AvgPool") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("AvgPool3D") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("LRN") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("Softplus") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("Softsign") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("FractionalAvgPool") { op, grad_inputs, grad_outputs ->
  
  }
  register_gradient_op("FractionalMaxPool") { op, grad_inputs, grad_outputs ->
  
  }
}