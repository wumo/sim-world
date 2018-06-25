package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*

class OperationBuilder(val graph: Graph, val type: String, val name: String) {
  lateinit var nativeOpDesc: tensorflow.TF_OperationDescription
  
  init {
    graph.ref().use {
      nativeOpDesc = TF_NewOperation(graph.nativeGraph, type, name)
    }
  }
  
  
  fun build(): Operation {
    graph.ref().use {
      val status = TF_NewStatus()
      val nativeOp = TF_FinishOperation(nativeOpDesc, status)
      val op = Operation(graph, nativeOp)
      TF_DeleteStatus(status)
      return op
    }
  }
  
  fun addInput(): OperationBuilder {
    graph.ref().use {
    
    }
    return this
  }
  
  fun setAttr(name: String, value: String): OperationBuilder {
    graph.ref().use {
      val value = value.toByteArray()
      TF_SetAttrString(nativeOpDesc, name, BytePointer(*value), value.size.toLong())
    }
    return this
  }
}