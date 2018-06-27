package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.tensorflow.*

class Operation(graph: Graph, val nativeOp: TF_Operation) {
  val name = graph.ref().use { TF_OperationName(nativeOp).string }
  val type = graph.ref().use { TF_OperationOpType(nativeOp).string }
  val numOutputs = graph.ref().use { TF_OperationNumOutputs(nativeOp) }
  operator fun get(idx: Int) = Output(this, idx)
}