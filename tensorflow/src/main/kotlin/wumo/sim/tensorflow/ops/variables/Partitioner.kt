package wumo.sim.tensorflow.ops.variables

import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape

interface Partitioner {
  operator fun invoke(dataType: DataType<*>, shape: Shape): IntArray
}