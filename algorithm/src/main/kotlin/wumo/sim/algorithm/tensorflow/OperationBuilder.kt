package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.util.Dimension
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.toByte

val String.fullName
  get() = substring(2)

fun TF.buildOp(op: String, name: String, setAttr: OperationBuilder.() -> Unit = {}) = run {
  name_scope(name) {
    val builder = g.nodeBuilder(op, ctxNs.scopeName.fullName)
    setAttr(builder)
    builder.build()
  }
}

fun TF.buildOpTensor(op: String, name: String, setAttr: OperationBuilder.() -> Unit = {}) = run {
  name_scope(name) {
    val builder = g.nodeBuilder(op, ctxNs.scopeName.fullName)
    setAttr(builder)
    val _op = builder.build()
    assert(_op.c_op.node().num_outputs() == 1) { "${_op.c_op.node().DebugString()} outputs > 1, use naryOps instead." }
    Tensor(_op, 0)
  }
}

fun TF.buildOpTensors(op: String, name: String, setAttr: OperationBuilder.() -> Unit = {}) = run {
  name_scope(name) {
    val builder = g.nodeBuilder(op, ctxNs.scopeName.fullName)
    setAttr(builder)
    val _op = builder.build()
    val outputs = _op.c_op.node().num_outputs()
    Array(outputs) { Tensor(_op, it) }
  }
}
//
//inline fun TF.buildOp(op: String, vararg inputs: Tensor, name: String, setAttr: OperationBuilder.() -> Unit = {}): Op {
//  name_scope(name) {
//    val builder = g.nodeBuilder(op, ctxNs.scopeName.fullName)
//    for (input in inputs)
//      builder.addInput(input)
//    setAttr(builder)
//    return builder.build()
//  }
//}
//
//inline fun TF.naryOp(op: String, vararg inputs: Tensor, name: String, setAttr: OperationBuilder.() -> Unit = {}): Tensor {
//  val _op = buildOp(op, *inputs, name = name, setAttr = setAttr)
//  assert(_op.c_op.node().num_outputs() == 1) { "${_op.c_op.node().DebugString()} outputs > 1, use naryOps instead." }
//  return Tensor(_op, 0)
//}
//
//inline fun TF.naryOps(op: String, vararg inputs: Tensor, name: String, setAttr: OperationBuilder.() -> Unit = {}): Array<Tensor> {
//  val _op = buildOp(op, *inputs, name = name, setAttr = setAttr)
//  val outputs = _op.c_op.node().num_outputs()
//  return Array(outputs) { Tensor(_op, it) }
//}
//
//inline fun TF.unaryOp(op: String, a: Tensor, name: String): Tensor {
//  name_scope(name) {
//    val op = g.nodeBuilder(op, ctxNs.scopeName.fullName).run {
//      addInput(a)
//      build()
//    }
//    return Tensor(op, 0)
//  }
//}
//
//inline fun TF.binaryOp(op: String, a: Tensor, b: Tensor, name: String): Tensor {
//  name_scope(name) {
//    val op = g.nodeBuilder(op, ctxNs.scopeName.fullName).run {
//      addInput(a)
//      addInput(b)
//      build()
//    }
//    return Tensor(op, 0)
//  }
//}
//
//inline fun TF.ternaryOp(op: String, a: Tensor, b: Tensor, c: Tensor, name: String): Tensor {
//  name_scope(name) {
//    val op = g.nodeBuilder(op, ctxNs.scopeName.fullName).run {
//      addInput(a)
//      addInput(b)
//      addInput(c)
//      build()
//    }
//    return Tensor(op, 0)
//  }
//}

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
  
  private fun addInput(input: TF_Output) {
    TF_AddInput(c_opDesc, input)
  }
  
  private val input_ops = mutableSetOf<Op>()
  private fun checkRef(input: Tensor, ref: Boolean) =
      if (ref) input.asRef() else input.value()
  
  fun addInput(input: Tensor, ref: Boolean = false) =
      addInput(checkRef(input, ref).asTF_Output())
          .apply { input_ops += input.op!! }
  
  fun addInput(input: Collection<Tensor>, ref: Boolean = false) {
    val inputs = TF_Output(input.size.toLong())
    for ((i, _input) in input.withIndex()) {
      val t = checkRef(_input, ref)
      inputs.position(i.toLong()).oper(t.op!!.c_op).index(t.value_index)
          .apply {
            input_ops += t.op
          }
    }
    TF_AddInputList(c_opDesc, inputs.position(0L), input.size)
  }
  
  fun addInput(input: Array<Tensor>, ref: Boolean = false) {
    val inputs = TF_Output(input.size.toLong())
    for ((i, _input) in input.withIndex()) {
      val t = checkRef(_input, ref)
      inputs.position(i.toLong()).oper(t.op!!.c_op).index(t.value_index)
          .apply {
            input_ops += t.op
          }
    }
    TF_AddInputList(c_opDesc, inputs.position(0L), input.size)
  }
  
  fun addControlInput(control: Op) {
    TF_AddControlInput(c_opDesc, control.c_op)
  }
  
  fun attr(name: String, value: String) {
    attr(name, value.toByteArray())
  }
  
  fun attr(name: String, values: Array<String>) {
    val attrValue = AttrValue()
    attrValue.mutable_list().apply {
      values.forEach { value ->
        add_s(value)
      }
    }
    attr(name, attrValue)
  }
  
  fun attr(name: String, value: ByteArray) {
    TF_SetAttrString(c_opDesc, name, BytePointer(*value), value.size.toLong())
  }
  
  fun attr(name: String, value: Boolean) {
    TF_SetAttrBool(c_opDesc, name, value.toByte())
  }
  
  fun attr(name: String, value: BooleanArray) {
    TF_SetAttrBoolList(c_opDesc, name,
                       ByteArray(value.size) { value[it].toByte() }, value.size)
  }
  
  fun attr(name: String, value: Int) {
    TF_SetAttrInt(c_opDesc, name, value.toLong())
  }
  
  fun attr(name: String, value: IntArray) {
    TF_SetAttrIntList(c_opDesc, name,
                      LongArray(value.size) { value[it].toLong() }, value.size)
  }
  
  fun attr(name: String, value: Long) {
    TF_SetAttrInt(c_opDesc, name, value)
  }
  
  fun attr(name: String, value: LongArray) {
    TF_SetAttrIntList(c_opDesc, name, value, value.size)
  }
  
  fun attr(name: String, value: Array<Long>) {
    TF_SetAttrIntList(c_opDesc, name, value.toLongArray(), value.size)
  }
  
  fun attr(name: String, value: Float) {
    TF_SetAttrFloat(c_opDesc, name, value)
  }
  
  fun attr(name: String, value: FloatArray) {
    TF_SetAttrFloatList(c_opDesc, name, value, value.size)
  }
  
  fun attr(name: String, value: Array<Float>) {
    TF_SetAttrFloatList(c_opDesc, name, value.toFloatArray(), value.size)
  }
  
  fun attrType(name: String, dtype: Int) {
    TF_SetAttrType(c_opDesc, name, dtype)
  }
  
  fun <T : Any> attr(name: String, value: TensorBuffer<T>) {
    val status = newStatus()
    TF_SetAttrTensor(c_opDesc, name, value.c_tensor, status)
    throwExceptionIfNotOk(status)
  }
  
  fun attr(name: String, value: Dimension) {
    TF_SetAttrShape(c_opDesc, name, value.asLongArray(), value.rank().toInt())
  }
  
  fun attr(name: String, value: NDArray<*>) {
    val status = newStatus()
    TF_SetAttrTensor(c_opDesc, name, TensorBuffer.fromNDArray(value).c_tensor, status)
    throwExceptionIfNotOk(status)
  }
  
  fun attr(name: String, value: TensorProto) {
    val attrValue = AttrValue()
    attrValue.mutable_tensor().CopyFrom(value)
    attr(name, attrValue)
  }
  
  fun attr(name: String, shapes: Array<Dimension>) {
    val attrValue = AttrValue()
    attrValue.mutable_list().apply {
      shapes.forEach { shape ->
        add_shape().apply {
          shape.forEach { dim ->
            add_dim().set_size(dim.toLong())
          }
        }
      }
    }
    attr(name, attrValue)
  }
  
  fun attr(name: String, func: NameAttrList) {
    val attrValue = AttrValue()
    attrValue.mutable_func().put(func)
    attr(name, attrValue)
  }
  
  fun attr(name: String, attrValue: AttrValue) {
    val status = newStatus()
    val buf = attrValue.SerializeAsString()
    TF_SetAttrValueProto(c_opDesc, name, buf, buf.limit(), status)
    throwExceptionIfNotOk(status)
  }
  
  fun attr(name: String, tensor_shape_proto: TensorShapeProto) {
    val status = newStatus()
    val buf = tensor_shape_proto.SerializeAsString()
    TF_SetAttrTensorShapeProto(c_opDesc, name, buf, buf.limit(), status)
    throwExceptionIfNotOk(status)
  }
  
  fun setDevice(device: String) {
    TF_SetDevice(c_opDesc, device)
  }
}