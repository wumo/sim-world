package wumo.sim.tensorflow.ops.variables

import wumo.sim.tensorflow.core.Graph
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape

class ResourceVariable:VariableLike {
  override val graph: Graph
    get() = TODO("not implemented")
  override val name: String
    get() = TODO("not implemented")
  override val dataType: DataType<*>
    get() = TODO("not implemented")
  override val shape: Shape
    get() = TODO("not implemented")
  override val value: Output
    get() = TODO("not implemented")
  override val initializer: Op
    get() = TODO("not implemented")
  override val isInitialized: Output
    get() = TODO("not implemented")
  override val initializedValue: Output
    get() = TODO("not implemented")
  
  override fun read(name: String): Output {
    TODO("not implemented")
  }
  
  override fun gather(indices: Output, name: String): Output {
    TODO("not implemented")
  }
  
  override fun assign(value: Output, name: String): Output {
    TODO("not implemented")
  }
  
  override fun assignAdd(value: Output, name: String): Output {
    TODO("not implemented")
  }
  
  override fun assignSub(value: Output, name: String): Output {
    TODO("not implemented")
  }
}