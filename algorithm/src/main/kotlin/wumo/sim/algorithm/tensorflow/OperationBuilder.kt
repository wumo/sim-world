package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.util.Dimension
import wumo.sim.util.toByte

val String.fullName
  get() = substring(2)

inline fun TF.buildOp(op: String, vararg inputs: Tensor, name: String, setAttr: OperationBuilder.() -> Unit = {}): Op {
  name_scope(name) {
    val builder = g.nodeBuilder(op, ctxNs.scopeName.fullName)
    for (input in inputs)
      builder.addInput(input)
    setAttr(builder)
    return builder.build()
  }
}

inline fun TF.naryOp(op: String, vararg inputs: Tensor, name: String, setAttr: OperationBuilder.() -> Unit = {}): Tensor {
  val _op = buildOp(op, *inputs, name = name, setAttr = setAttr)
  assert(_op.c_op.node().num_outputs() == 1) { "${_op.c_op.node().DebugString()} outputs > 1, use naryOps instead." }
  return Tensor(_op, 0)
}

inline fun TF.naryOps(op: String, vararg inputs: Tensor, name: String, setAttr: OperationBuilder.() -> Unit = {}): Array<Tensor> {
  val _op = buildOp(op, *inputs, name = name, setAttr = setAttr)
  val outputs = _op.c_op.node().num_outputs()
  return Array(outputs) { Tensor(_op, it) }
}

inline fun TF.unaryOp(op: String, a: Tensor, name: String): Tensor {
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
  
  fun build(): Op {
    controlInput()
    attr_scope()
    colocate()
    val status = newStatus()
    val nativeOp = TF_FinishOperation(c_opDesc, status)
    throwExceptionIfNotOk(status)
    val op = Op(graph, nativeOp)
    op.control_flow_context = tf.control_flow_context
    
    control_flow_post_processing(op)
    return op
  }
  
  private fun colocate() {
    val all_colocation_groups = hashSetOf<String>()
    for (op in tf.colocate_with) {
      val class_attr = op.attrStringList("_class")
      var hasDependent = false
      if (class_attr != null) {
        for (class_name in class_attr)
          if (class_name.startsWith("loc:@")) {
            all_colocation_groups += class_name
            hasDependent = true
          }
      }
      if (!hasDependent)
        all_colocation_groups += "loc:@${op.name}"
    }
    if (all_colocation_groups.isEmpty()) return
    val a = AttrValue()
    a.mutable_list().apply {
      for (loc in all_colocation_groups)
        add_s(loc)
    }
    attr("_class", a)
  }
  
  private fun attr_scope() {
    tf.attr_scope_map.forEach { key, value ->
      attr(key, value)
    }
  }
  
  private fun control_flow_post_processing(op: Op) {
    op.control_flow_context?.addOp(op)
  }
  
  private fun controlInput() {
    for (control_op in tf.control_ops) {
      //If any of the input_ops already depends on the inputs from controller,
      //we say that the new op is dominated (by that input), and we therefore
      //do not need to add control dependencies for this controller's inputs.
      var dominated = false
      for (input_op in input_ops)
        if (input_op.control_inputs.contains(control_op)) {
          dominated = true
          break
        }
      if (!dominated) {
        //Don't add a control input if we already have a data dependency on it.
        //NOTE(mrry): We do not currently track transitive data dependencies,
        //so we may add redundant control inputs.
        addControlInput(control_op)
      }
    }
  }
  
  private fun addInput(input: TF_Output): OperationBuilder {
    TF_AddInput(c_opDesc, input)
    return this
  }
  
  private val input_ops = mutableSetOf<Op>()
  
  fun addInput(input: Tensor) = addInput(input.asTF_Output()).apply { input_ops += input.op!! }
  
  fun addInputList(input: Collection<Tensor>): OperationBuilder {
    val inputs = TF_Output(input.size.toLong())
    for ((i, _input) in input.withIndex())
      inputs.position(i.toLong()).oper(_input.op!!.c_op).index(_input.value_index)
          .apply {
            input_ops += _input.op
          }
    TF_AddInputList(c_opDesc, inputs.position(0L), input.size)
    return this
  }
  
  fun addControlInput(control: Op): OperationBuilder {
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