package wumo.sim.tensorflow.layers

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.types.DataType

//
//import org.bytedeco.javacpp.tensorflow.DT_INVALID
//import wumo.sim.tensorflow.ops.Output
//import wumo.sim.tensorflow.ops.variables.Initializer
//import wumo.sim.tensorflow.tf
//import wumo.sim.tensorflow.types.DataType
//import wumo.sim.util.Shape
//
//typealias TensorFunction = (Output) -> Output?
//
open class Layer(val trainable: Boolean = true,
                 val name: String? = null,
                 val dataType: DataType<*>? = null,
                 val activity_reqularizer: Any? = null) {
  
  //
  var statefule = false
  var built = false
  val losses = mutableListOf<Output>()
//
//  open fun build(input_shape: Shape) {
//    built = true
//  }
//
//  open fun call(input: Output) = input
//
//  open operator fun invoke(inputs: Output): Output {
//    if (!built) {
//      if (dataType == DT_INVALID)
//        dataType = inputs.dataType
//      val input_shape = inputs.shape
//      build(input_shape)
//    }
//    //if not in_deferred_mode:
//    val outputs = call(inputs)
//    if (activity_reqularizer != null) {
//    }
//    return outputs
//  }
//
//  protected fun add_variable(shape: Shape, dataType: DataType<*>? = null,
//                             initializer: Initializer,
//                             regularizer: TensorFunction? = null,
//                             trainable: Boolean = true,
//                             name: String = ""): Output {
//    val v = tf.variable(name, shape, dataType, initializer, trainable = trainable && this.trainable)
//    if (regularizer != null)
//      losses += regularizer(v)
//    return v
//  }
}