package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.helper.tensorflow
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.Dimension

class OperationBuilder(val graph: Graph, val opType: String, val name: String) {
  private var _opDescription: TF_OperationDescription = TF_NewOperation(graph.c_graph, opType, name)
  
  fun build(): Operation {
    val status = newStatus()
    val nativeOp = TF_FinishOperation(_opDescription, status)
    throwExceptionIfNotOk(status)
    val op = Operation(graph, nativeOp)
    return op
  }
  
  fun addInput(input: Tensor): OperationBuilder {
    TF_AddInput(_opDescription, input.asTF_Output())
    return this
  }
  
  fun addControlInput(control: Operation): OperationBuilder {
    TF_AddControlInput(_opDescription, control.c_op)
    return this
  }
  
  fun setAttr(name: String, value: String): OperationBuilder {
    return setAttr(name, value.toByteArray())
  }
  
  fun setAttr(name: String, value: ByteArray): OperationBuilder {
    TF_SetAttrString(_opDescription, name, BytePointer(*value), value.size.toLong())
    return this
  }
  
  fun setAttr(name: String, value: Int): OperationBuilder {
    TF_SetAttrInt(_opDescription, name, value.toLong())
    return this
  }
  
  fun setAttr(name: String, value: Float): OperationBuilder {
    TF_SetAttrFloat(_opDescription, name, value)
    return this
  }
  
  fun setAttrType(name: String, dtype: Int): OperationBuilder {
    TF_SetAttrType(_opDescription, name, dtype)
    return this
  }
  
  fun <T> setAttr(name: String, value: TensorValue<T>): OperationBuilder {
    val status = TF_NewStatus()
    TF_SetAttrTensor(_opDescription, name, value.c_tensor, status)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
    return this
  }
  
  fun setAttr(name: String, value: Dimension): OperationBuilder {
    TF_SetAttrShape(_opDescription, name, value.asLongArray(), value.rank().toInt())
    return this
  }
  
  fun setAttr(name: String, attrValue: AttrValue): OperationBuilder {
    val status = TF_NewStatus()
    val buf = attrValue.SerializeAsString()
    TF_SetAttrValueProto(_opDescription, name, buf, buf.limit(), status)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
    return this
  }
}