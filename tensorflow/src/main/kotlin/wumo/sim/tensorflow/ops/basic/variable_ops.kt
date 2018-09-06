package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_variable_ops
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape

object variable_ops {
  interface API {
    fun zeroInitializer(_ref: Output, name: String = "ZeroInitializer"): Output {
      return gen_variable_ops.zeroInitializer(_ref, name)
    }
    
    fun zeroVarInitializer(_var: Output, dtype: DataType<*>, shape: Shape, name: String = "ZeroVarInitializer"): Output {
      return gen_variable_ops.zeroVarInitializer(_var, dtype, shape, name)
    }
  }
}