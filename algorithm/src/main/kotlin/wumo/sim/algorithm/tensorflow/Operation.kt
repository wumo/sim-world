package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.tensorflow.*

class Operation(val graph: Graph, val c_op: TF_Operation) {
  val name = TF_OperationName(c_op).string
}