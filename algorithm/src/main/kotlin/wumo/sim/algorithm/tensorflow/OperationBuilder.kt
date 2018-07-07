package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.Dimension
import wumo.sim.algorithm.util.helpers.toByte

class OperationBuilder(val graph: Graph, val opType: String, val name: String) {
  private var c_opDesc: TF_OperationDescription = TF_NewOperation(graph.c_graph, opType, name)
  
  fun build(): Operation {
    val status = newStatus()
    val nativeOp = TF_FinishOperation(c_opDesc, status)
    throwExceptionIfNotOk(status)
    val op = Operation(graph, nativeOp)
    return op
  }
  
  fun addInput(input: TF_Output): OperationBuilder {
    TF_AddInput(c_opDesc, input)
    return this
  }
  
  fun addInput(input: Tensor) = addInput(input.asTF_Output())
  
  fun addInputList(input: Array<TF_Output>): OperationBuilder {
    val inputs = TF_Output(input.size.toLong())
    for ((i, input) in input.withIndex())
      inputs.position(i.toLong()).oper(input.oper()).index(input.index())
    TF_AddInputList(c_opDesc, inputs.position(0L), input.size)
    return this
  }
  
  fun addInputList(input: Array<Tensor>): OperationBuilder {
    val inputs = TF_Output(input.size.toLong())
    for ((i, input) in input.withIndex())
      inputs.position(i.toLong()).oper(input.op.c_op).index(input.value_index)
    TF_AddInputList(c_opDesc, inputs.position(0L), input.size)
    return this
  }
  
  fun addControlInput(control: Operation): OperationBuilder {
    TF_AddControlInput(c_opDesc, control.c_op)
    return this
  }
  
  fun setAttr(name: String, value: String): OperationBuilder {
    return setAttr(name, value.toByteArray())
  }
  
  fun setAttr(name: String, value: ByteArray): OperationBuilder {
    TF_SetAttrString(c_opDesc, name, BytePointer(*value), value.size.toLong())
    return this
  }
  
  fun setAttr(name: String, value: Boolean): OperationBuilder {
    TF_SetAttrBool(c_opDesc, name, value.toByte())
    return this
  }
  
  fun setAttr(name: String, value: BooleanArray): OperationBuilder {
    TF_SetAttrBoolList(c_opDesc, name,
                       ByteArray(value.size) { value[it].toByte() }, value.size)
    return this
  }
  
  fun setAttr(name: String, value: Int): OperationBuilder {
    TF_SetAttrInt(c_opDesc, name, value.toLong())
    return this
  }
  
  fun setAttr(name: String, value: IntArray): OperationBuilder {
    TF_SetAttrIntList(c_opDesc, name,
                      LongArray(value.size) { value[it].toLong() }, value.size)
    return this
  }
  
  fun setAttr(name: String, value: Float): OperationBuilder {
    TF_SetAttrFloat(c_opDesc, name, value)
    return this
  }
  
  fun setAttr(name: String, value: FloatArray): OperationBuilder {
    TF_SetAttrFloatList(c_opDesc, name, value, value.size)
    return this
  }
  
  fun setAttrType(name: String, dtype: Int): OperationBuilder {
    TF_SetAttrType(c_opDesc, name, dtype)
    return this
  }
  
  fun <T> setAttr(name: String, value: TensorValue<T>): OperationBuilder {
    val status = TF_NewStatus()
    TF_SetAttrTensor(c_opDesc, name, value.c_tensor, status)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
    return this
  }
  
  fun setAttr(name: String, value: Dimension): OperationBuilder {
    TF_SetAttrShape(c_opDesc, name, value.asLongArray(), value.rank().toInt())
    return this
  }
  
  fun setAttr(name: String, attrValue: AttrValue): OperationBuilder {
    val status = TF_NewStatus()
    val buf = attrValue.SerializeAsString()
    TF_SetAttrValueProto(c_opDesc, name, buf, buf.limit(), status)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
    return this
  }
}