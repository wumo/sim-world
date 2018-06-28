package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.tensorflow.*
import org.tensorflow.framework.DataType
import wumo.sim.algorithm.util.Dimension

class OperationBuilder(val graph: Graph, val type: String, val name: String) {
  lateinit var nativeOpDesc: TF_OperationDescription
  
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
  
  fun addInput(op: Operation) = addInput(op[0])
  
  fun addInput(input: Output): OperationBuilder {
    graph.ref().use {
      val out = TF_Output().oper(input.op.nativeOp).index(input.idx)
      TF_AddInput(nativeOpDesc, out)
    }
    return this
  }
  
  fun addControlInput(control: Operation): OperationBuilder {
    graph.ref().use {
      TF_AddControlInput(nativeOpDesc, control.nativeOp)
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
  
  fun setAttr(name: String, value: Int): OperationBuilder {
    graph.ref().use {
      TF_SetAttrInt(nativeOpDesc, name, value.toLong())
    }
    return this
  }
  
  fun setAttr(name: String, value: Float): OperationBuilder {
    graph.ref().use {
      TF_SetAttrFloat(nativeOpDesc, name, value)
    }
    return this
  }
  
  fun setAttr(name: String, value: DataType): OperationBuilder {
    graph.ref().use {
      TF_SetAttrType(nativeOpDesc, name, value.number)
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
  
  fun setAttr(name: String, value: Dimension): OperationBuilder {
    graph.ref().use {
      TF_SetAttrShape(nativeOpDesc, name, value.asLongArray(), value.rank().toInt())
    }
    return this
  }
  
  fun setAttr(name: String, attrValue: AttrValue): OperationBuilder {
    graph.ref().use {
      val status = TF_NewStatus()
      val buf = attrValue.SerializeAsString()
      TF_SetAttrValueProto(nativeOpDesc, name, buf, buf.limit(), status)
      throwExceptionIfNotOk(status)
      TF_DeleteStatus(status)
    }
    return this
  }
}