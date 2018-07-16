package wumo.sim.algorithm.drl.deepq

import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.contrib.fully_connected
import wumo.sim.algorithm.tensorflow.layers.layer_norm
import wumo.sim.algorithm.tensorflow.ops.relu
import wumo.sim.algorithm.tensorflow.tf

typealias Q_func = (Tensor, Int, String) -> Tensor

fun TF._mlp(hiddens: IntArray, input: Tensor, num_actions: Int, layer_norm: Boolean = false, name: String = "mlp"): Tensor {
  tf.subscope(name) {
    var out = input
    for (hidden in hiddens) {
      fully_connected(out, num_outputs = hidden, activation_fn = null)
      if (layer_norm)
        out = layer_norm(out, center = true, scale = true)
      out = relu(out)
    }
    val q_out = fully_connected(out, num_outputs = num_actions, activation_fn = null)
    return q_out
  }
}

fun TF.mlp(vararg hiddens: Int, layer_norm: Boolean = false): Q_func {
  return { input, num_outputs, name ->
    _mlp(hiddens, input, num_outputs, layer_norm, name)
  }
}