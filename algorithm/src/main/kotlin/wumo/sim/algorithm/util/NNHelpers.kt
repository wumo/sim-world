package wumo.sim.algorithm.util

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.tensorflow.*


class TFHelper(val scope: Scope) {
  val trainables = mutableListOf<Variable>()
  
  fun placeholder(name: String, shape: Dimension) =
      Placeholder(scope.WithOpName(name), DT_FLOAT,
          Placeholder.Shape(TensorShape(*shape.asLongArray()).asPartialTensorShape()))
  
  fun variable(name: String, shape: Dimension, trainable: Boolean = true) =
      Variable(scope.WithOpName(name),
          TensorShape(*shape.asLongArray()).asPartialTensorShape(), DT_FLOAT)
          .apply {
            if (trainable)
              trainables += this
          }
  
  fun random_uniform(shape: Dimension): RandomUniform {
    return RandomUniform(scope,
        Input(Tensor.create(shape.asLongArray(),
            TensorShape(*longArrayOf(shape.rank())))), DT_FLOAT)
  }
  
  fun const(data: Boolean) = Const(scope, data)
  fun const(data: Byte) = Const(scope, data)
  fun const(data: Short) = Const(scope, data)
  fun const(data: Int) = Const(scope, data)
  fun const(data: Float) = Const(scope, data)
  fun const(data: Long) = Const(scope, data)
  fun const(data: Double) = Const(scope, data)
}