package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_collective_ops
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape

object collective_ops {
  interface API {
    fun collectiveBcastRecv(t: DataType<*>, groupSize: Long, groupKey: Long, instanceKey: Long, shape: Shape, name: String = "CollectiveBcastRecv"): Output {
      return gen_collective_ops.collectiveBcastRecv(t, groupSize, groupKey, instanceKey, shape, name)
    }
    
    fun collectiveBcastSend(input: Output, groupSize: Long, groupKey: Long, instanceKey: Long, shape: Shape, name: String = "CollectiveBcastSend"): Output {
      return gen_collective_ops.collectiveBcastSend(input, groupSize, groupKey, instanceKey, shape, name)
    }
    
    fun collectiveReduce(input: Output, groupSize: Long, groupKey: Long, instanceKey: Long, mergeOp: String, finalOp: String, subdivOffsets: Array<Long>, name: String = "CollectiveReduce"): Output {
      return gen_collective_ops.collectiveReduce(input, groupSize, groupKey, instanceKey, mergeOp, finalOp, subdivOffsets, name)
    }
  }
}