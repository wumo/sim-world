package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.util.Dimension
import wumo.sim.util.toByte

val String.fullName
  get() = substring(2)

inline fun TF.naryOp(op: String, vararg inputs: Tensor, name: String, setAttr: OperationBuilder.() -> Unit = {}): Tensor {
  name_scope(name) {
    val builder = g.nodeBuilder(op, ctxNs.scopeName.fullName)
    for (input in inputs)
      builder.addInput(input)
    setAttr(builder)
    val op = builder.build()
    assert(op.c_op.node().num_outputs() == 1) { "${op.c_op.node().DebugString()} outputs > 1, use naryOps instead." }
    return Tensor(op, 0)
  }
}

inline fun TF.naryOps(op: String, vararg inputs: Tensor, name: String, setAttr: OperationBuilder.() -> Unit = {}): Array<Tensor> {
  name_scope(name) {
    val builder = g.nodeBuilder(op, ctxNs.scopeName.fullName)
    for (input in inputs)
      builder.addInput(input)
    setAttr(builder)
    val op = builder.build()
    val outputs = op.c_op.node().num_outputs()
    return Array(outputs) { Tensor(op, it) }
  }
}

inline fun TF.unaryOp(op: String, a: Tensor, name: String, dtype: Int = a.dtype): Tensor {
  name_scope(name) {
    val op = g.nodeBuilder(op, ctxNs.scopeName.fullName)
        .addInput(a)
        .build()
    return Tensor(op, 0)
  }
}

inline fun TF.binaryOp(op: String, a: Tensor, b: Tensor, name: String): Tensor {
  name_scope(name) {
    val op = g.nodeBuilder(op, ctxNs.scopeName.fullName)
        .addInput(a)
        .addInput(b)
        .build()
    return Tensor(op, 0)
  }
}

inline fun TF.ternaryOp(op: String, a: Tensor, b: Tensor, c: Tensor, name: String): Tensor {
  name_scope(name) {
    val op = g.nodeBuilder(op, ctxNs.scopeName.fullName)
        .addInput(a)
        .addInput(b)
        .addInput(c)
        .build()
    return Tensor(op, 0)
  }
}

class OperationBuilder(val graph: Graph, val opType: String, val name: String) {
  private var c_opDesc: TF_OperationDescription = TF_NewOperation(graph.c_graph, opType, name)
  
  fun build(): Operation {
    tf.ctxNs.colocate_with?.apply {
      TF_ColocateWith(c_opDesc, this.c_op)
    }
    addContextControlInput()
    tf.attr_scope_map.forEach { key, value ->
      attr(key, value)
    }
    val status = newStatus()
    val nativeOp = TF_FinishOperation(c_opDesc, status)
    throwExceptionIfNotOk(status)
    val op = Operation(graph, nativeOp)
    control_flow_post_processing(op)
//    tf.attr_scope_map.forEach { key, value ->
//      op.set_attr(key, value)
//    }
    return op
  }
  
  private fun control_flow_post_processing(op: Operation) {
    tf.control_flow_context?.addOp(op)
  }
  
  private fun addContextControlInput(): OperationBuilder {
    for (control_op in tf.ctxNs.control_ops)
      addControlInput(control_op)
    return this
  }
  
  fun addInput(input: TF_Output): OperationBuilder {
    TF_AddInput(c_opDesc, input)
    return this
  }
  
  fun addInput(input: Tensor) = addInput(input.asTF_Output())
  
  fun addInputList(input: Collection<Tensor>): OperationBuilder {
    val inputs = TF_Output(input.size.toLong())
    for ((i, _input) in input.withIndex())
      inputs.position(i.toLong()).oper(_input.op!!.c_op).index(_input.value_index)
    TF_AddInputList(c_opDesc, inputs.position(0L), input.size)
    return this
  }
  
  fun addInputList(input: Array<Tensor>): OperationBuilder {
    val inputs = TF_Output(input.size.toLong())
    for ((i, _input) in input.withIndex())
      inputs.position(i.toLong()).oper(_input.op!!.c_op).index(_input.value_index)
    TF_AddInputList(c_opDesc, inputs.position(0L), input.size)
    return this
  }
  
  fun addControlInput(control: Operation): OperationBuilder {
    TF_AddControlInput(c_opDesc, control.c_op)
    return this
  }
  
  fun attr(name: String, value: String): OperationBuilder {
    return attr(name, value.toByteArray())
  }
  
  fun attr(name: String, value: ByteArray): OperationBuilder {
    TF_SetAttrString(c_opDesc, name, BytePointer(*value), value.size.toLong())
    return this
  }
  
  fun attr(name: String, value: Boolean): OperationBuilder {
    TF_SetAttrBool(c_opDesc, name, value.toByte())
    return this
  }
  
  fun attr(name: String, value: BooleanArray): OperationBuilder {
    TF_SetAttrBoolList(c_opDesc, name,
                       ByteArray(value.size) { value[it].toByte() }, value.size)
    return this
  }
  
  fun attr(name: String, value: Int): OperationBuilder {
    TF_SetAttrInt(c_opDesc, name, value.toLong())
    return this
  }
  
  fun attr(name: String, value: IntArray): OperationBuilder {
    TF_SetAttrIntList(c_opDesc, name,
                      LongArray(value.size) { value[it].toLong() }, value.size)
    return this
  }
  
  fun attr(name: String, value: Long): OperationBuilder {
    TF_SetAttrInt(c_opDesc, name, value)
    return this
  }
  
  fun attr(name: String, value: LongArray): OperationBuilder {
    TF_SetAttrIntList(c_opDesc, name, value, value.size)
    return this
  }
  
  fun attr(name: String, value: Float): OperationBuilder {
    TF_SetAttrFloat(c_opDesc, name, value)
    return this
  }
  
  fun attr(name: String, value: FloatArray): OperationBuilder {
    TF_SetAttrFloatList(c_opDesc, name, value, value.size)
    return this
  }
  
  fun attrType(name: String, dtype: Int): OperationBuilder {
    TF_SetAttrType(c_opDesc, name, dtype)
    return this
  }
  
  fun <T : Any> attr(name: String, value: TensorBuffer<T>): OperationBuilder {
    val status = newStatus()
    TF_SetAttrTensor(c_opDesc, name, value.c_tensor, status)
    throwExceptionIfNotOk(status)
    return this
  }
  
  fun attr(name: String, value: Dimension): OperationBuilder {
    TF_SetAttrShape(c_opDesc, name, value.asLongArray(), value.rank().toInt())
    return this
  }
  
  fun attr(name: String, attrValue: AttrValue): OperationBuilder {
    val status = newStatus()
    val buf = attrValue.SerializeAsString()
    TF_SetAttrValueProto(c_opDesc, name, buf, buf.limit(), status)
    throwExceptionIfNotOk(status)
    return this
  }
  
  fun attr(name: String, tensor_shape_proto: TensorShapeProto): OperationBuilder {
    val status = newStatus()
    val buf = tensor_shape_proto.SerializeAsString()
    TF_SetAttrTensorShapeProto(c_opDesc, name, buf, buf.limit(), status)
    throwExceptionIfNotOk(status)
    return this
  }
  
  fun setDevice(device: String): OperationBuilder {
    TF_SetDevice(c_opDesc, device)
    return this
  }
}