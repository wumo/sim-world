package wumo.sim.tensorflow.ops.training

import wumo.sim.tensorflow.ops.IndexedSlices
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.times
import wumo.sim.tensorflow.ops.variables.Variable
import wumo.sim.tensorflow.tf

class GradientDescentOptimizer(
    val learningRate: () -> Double,
    override val useLocking: Boolean = false,
    override val name: String = "GradientDescent"
) : Optimizer() {
  
  override val ignoreDuplicateSparseIndices = true
  
  lateinit var learningRateTensor: Output
  
  protected fun getLearningRate(variable: Variable, iteration: Variable?): Output {
    if (!::learningRateTensor.isInitialized)
      throw IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    
    return tf.cast(learningRateTensor, variable.dataType)
  }
  
  override fun applyDense(gradient: Output, variable: Variable, iteration: Variable?): Op =
      tf.applyGradientDescent(variable.variable,
                               getLearningRate(variable, iteration),
                               gradient, useLocking).op!!
  
  override fun applySparseDuplicateIndices(gradient: IndexedSlices,
                                           variable: Variable,
                                           iteration: Variable?): Op =
      variable.assignScatterSub(gradient.indices,
                                gradient.values *
                                    getLearningRate(variable, iteration), useLocking).op!!
  
  override fun prepare(iteration: Variable?) {
    val _learningRate = learningRate()
    learningRateTensor = tf.const(_learningRate, name = "learning_rate")
  }
}

//
//import wumo.sim.tensorflow.ops.Output
//import wumo.sim.tensorflow.ops.variables.Variable
//import wumo.sim.tensorflow.ops.cast
//import wumo.sim.tensorflow.ops.const
//import wumo.sim.tensorflow.ops.gen.applyGradientDescent
//import wumo.sim.tensorflow.tf
//
//class GradientDescentOptimizer(val learningRate: Float,
//                               use_locking: Boolean = false,
//                               name: String = "GradientDescent") : Optimizer(use_locking, name) {
//  lateinit var lr_t: Output
//  override fun prepare() {
//    lr_t = tf.const(learningRate, name = "learning_rate")
//  }
//
//  override fun apply_dense(grad: Output, v: Variable) =
//      tf.applyGradientDescent(v, tf.cast(lr_t, v.dataType.base_dtype), grad).op!!
//
//}