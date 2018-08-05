package wumo.sim.algorithm.tensorflow.training

import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.Variable
import wumo.sim.algorithm.tensorflow.base_dtype
import wumo.sim.algorithm.tensorflow.ops.cast
import wumo.sim.algorithm.tensorflow.ops.const
import wumo.sim.algorithm.tensorflow.ops.gen.applyGradientDescent
import wumo.sim.algorithm.tensorflow.tf

class GradientDescentOptimizer(val learningRate: Float,
                               use_locking: Boolean = false,
                               name: String = "GradientDescent") : Optimizer(use_locking, name) {
  lateinit var lr_t: Tensor
  override fun prepare() {
    lr_t = tf.const(learningRate, name = "learning_rate")
  }
  
  override fun apply_dense(grad: Tensor, v: Variable) =
      tf.applyGradientDescent(v, tf.cast(lr_t, v.dtype.base_dtype), grad).op!!
  
}