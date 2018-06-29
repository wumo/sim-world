package wumo.sim.algorithm.util.cpp_api.ops

import org.bytedeco.javacpp.tensorflow.*
import org.bytedeco.javacpp.tensorflow.DT_FLOAT
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.cpp_api.MakeShape
import wumo.sim.algorithm.util.cpp_api.TF_CPP

fun TF_CPP.placeholder(shape: Dimension, dtype: Int = DT_FLOAT,
                       name: String = "",
                       scope: Scope = root): Output {
  return Placeholder(scope.WithOpName(name), dtype,
      Placeholder.Shape(MakeShape(shape.asLongArray())
          .asPartialTensorShape())).asOutput()
}