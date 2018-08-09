package wumo.sim.tensorflow.training

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.Variable
import wumo.sim.tensorflow.ops.cast
import wumo.sim.tensorflow.ops.const
import wumo.sim.tensorflow.ops.gen.applyGradientDescent
import wumo.sim.tensorflow.tf

class GradientDescentOptimizer(val learningRate: Float,
                               use_locking: Boolean = false,
                               name: String = "GradientDescent") : Optimizer(use_locking, name) {
  lateinit var lr_t: Output
  override fun prepare() {
    lr_t = tf.const(learningRate, name = "learning_rate")
  }
  
  override fun apply_dense(grad: Output, v: Variable) =
      tf.applyGradientDescent(v, tf.cast(lr_t, v.dtype.base_dtype), grad).op!!
  
}