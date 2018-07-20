package wumo.sim.algorithm.drl.deepq

import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.contrib.fully_connected
import wumo.sim.algorithm.tensorflow.layers.layer_norm
import wumo.sim.algorithm.tensorflow.ops.relu
import wumo.sim.algorithm.tensorflow.tf

typealias Q_func = (Tensor, Int, String, Boolean) -> Tensor

fun _mlp(hiddens: IntArray, input: Tensor, num_actions: Int, layer_norm: Boolean = false, name: String = "mlp", reuse: Boolean = false): Tensor {
  tf.variable_scope(name, reuse = reuse) {
    var out = input
    for (hidden in hiddens) {
      out = tf.fully_connected(out, num_outputs = hidden, activation_fn = null)
      if (layer_norm)
        out = tf.layer_norm(out, center = true, scale = true)
      out = tf.relu(out)
    }
    val q_out = tf.fully_connected(out, num_outputs = num_actions, activation_fn = null)
    return q_out
  }
}

fun mlp(vararg hiddens: Int, layer_norm: Boolean = false): Q_func {
  return { input, num_outputs, name, reuse ->
    _mlp(hiddens, input, num_outputs, layer_norm, name, reuse)
  }
}