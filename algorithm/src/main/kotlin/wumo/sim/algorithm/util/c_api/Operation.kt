package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.tensorflow.*
import org.tensorflow.framework.DataType

class Operation(val graph: Graph, val nativeOp: TF_Operation) {
  val name = graph.ref().use { TF_OperationName(nativeOp).string }
  val opType = graph.ref().use { TF_OperationOpType(nativeOp).string }
  val numOutputs = graph.ref().use { TF_OperationNumOutputs(nativeOp) }
  
  operator fun get(idx: Int) = Output(this, idx)
  
  fun outputType(idx: Int) = graph.ref().use {
    DataType.forNumber(TF_OperationOutputType(TF_Output().oper(nativeOp).index(idx)) % 100)//依据对应关系得出的公式，可能会变
  }
  
  fun inputType(idx: Int) = graph.ref().use {
    DataType.forNumber(TF_OperationInputType(TF_Input().oper(nativeOp).index(idx)) % 100)
  }
  
}