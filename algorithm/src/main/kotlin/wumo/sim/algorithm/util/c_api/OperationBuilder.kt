package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import org.tensorflow.TensorFlowException

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
      throwExceptionIfNotOk(status)
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
    return setAttr(name, value.toByteArray())
  }
  
  fun setAttr(name: String, value: ByteArray): OperationBuilder {
    graph.ref().use {
      TF_SetAttrString(nativeOpDesc, name, BytePointer(*value), value.size.toLong())
    }
    return this
  }
  
  fun setAttr(name: String, value: DataType): OperationBuilder {
    graph.ref().use {
      TF_SetAttrType(nativeOpDesc, name, value.c())
    }
    return this
  }
  
  fun setAttr(name: String, value: Tensor): OperationBuilder {
    graph.ref().use {
      val status = TF_NewStatus()
      TF_SetAttrTensor(nativeOpDesc, name, value.nativeTensor, status)
      throwExceptionIfNotOk(status)
      TF_DeleteStatus(status)
    }
    return this
  }
}