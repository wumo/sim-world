package wumo.sim.algorithm.drl.deepq

import wumo.sim.algorithm.tensorflow.ops.Output
import wumo.sim.algorithm.tensorflow.contrib.fully_connected
import wumo.sim.algorithm.tensorflow.contrib.layer_norm
import wumo.sim.algorithm.tensorflow.ops.gen.relu
import wumo.sim.algorithm.tensorflow.tf

typealias Q_func = (Output, Int, String, Boolean) -> Output

fun _mlp(hiddens: IntArray, input: Output, num_actions: Int, layer_norm: Boolean = false, name: String = "mlp", reuse: Boolean = false): Output {
  tf.variable_scope(name) {
    tf.ctxVs.reuse = reuse
    tf.ctxVs.reenter_increment = true
    var out = input
    for (hidden in hiddens) {
      out = tf.fully_connected(out, num_outputs = hidden, activation_fn = null)
      if (layer_norm)
        out = tf.layer_norm(out, center = true, scale = true)
      out = tf.relu(out)
    }
    val q_out = tf.fully_connected(out, num_outputs = num_actions, activation_fn = null)
    tf.ctxVs.reenter_increment = false
    return q_out
  }
}

fun mlp(vararg hiddens: Int, layer_norm: Boolean = false): Q_func {
  return { input, num_outputs, name, reuse ->
    _mlp(hiddens, input, num_outputs, layer_norm, name, reuse)
  }
}