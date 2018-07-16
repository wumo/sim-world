package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.tf


fun TF.truncatedNormal(shape: Tensor, dtype: Int = DT_FLOAT, name: String = "truncated_normal"): Tensor {
  val op = g.nodeBuilder("TruncatedNormal", ctx.getUniqueFullName(name))
      .addInput(shape)
      .setAttrType("dtype", dtype)
      .build()
  return Tensor(op, 0)
}

fun TF.truncatedNormal(shape: Tensor, mean: Float = 0f, stddev: Float = 1f, dtype: Int = DT_FLOAT, name: String = "truncated_normal"): Tensor {
  tf.subscope(name) {
    val mean_t = tf.const(mean, name = "mean")
    val stddev_t = tf.const(stddev, name = "stddev")
    val rnd = tf.truncatedNormal(shape, dtype, name)
    val mul = rnd * stddev_t
    return tf.add(mul, mean_t, name = name)
  }
}