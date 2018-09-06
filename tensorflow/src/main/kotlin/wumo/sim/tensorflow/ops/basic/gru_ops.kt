package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_gru_ops

object gru_ops {
  interface API {
    fun gRUBlockCell(x: Output, hPrev: Output, wRu: Output, wC: Output, bRu: Output, bC: Output, name: String = "GRUBlockCell"): List<Output> {
      return gen_gru_ops.gRUBlockCell(x, hPrev, wRu, wC, bRu, bC, name)
    }
    
    fun gRUBlockCellGrad(x: Output, hPrev: Output, wRu: Output, wC: Output, bRu: Output, bC: Output, r: Output, u: Output, c: Output, dH: Output, name: String = "GRUBlockCellGrad"): List<Output> {
      return gen_gru_ops.gRUBlockCellGrad(x, hPrev, wRu, wC, bRu, bC, r, u, c, dH, name)
    }
  }
}