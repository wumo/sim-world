package wumo.sim.tensorflow

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import org.tensorflow.framework.OpDef
import wumo.sim.tensorflow.core.check
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.control_flow_ops.control_flow_ops.checkInputFromValidContext
import wumo.sim.tensorflow.ops.ops
import wumo.sim.tensorflow.ops.ops.graphConstructionScope
import wumo.sim.tensorflow.ops.ops.logger
import wumo.sim.tensorflow.ops.ops.pruneControlDependencies
import wumo.sim.tensorflow.tensor.Tensor
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.toByte
import wumo.sim.util.warn
import java.util.*
import java.util.Collections.emptySet as emptyMutableSet

fun buildOp(op: String, name: String, setAttr: OperationBuilder.() -> Unit = {}) = run {
  tf.nameScope(name) {
    val builder = tf.currentGraph.nodeBuilder(op, ops.convertNameScopeToName(tf.currentNameScope))
    setAttr(builder)
    builder.build()
  }
}

fun buildOpTensor(op: String, name: String, setAttr: OperationBuilder.() -> Unit = {}) = run {
  tf.nameScope(name) {
    val builder = tf.currentGraph.nodeBuilder(op, ops.convertNameScopeToName(tf.currentNameScope))
    setAttr(builder)
    val _op = builder.build()
    assert(_op.c_op.node().num_outputs() == 1) { "${_op.c_op.node().DebugString()} outputs > 1, use naryOps instead." }
    Output(_op, 0)
  }
}

fun buildOpTensors(op: String, name: String, setAttr: OperationBuilder.() -> Unit = {}) = run {
  tf.nameScope(name) {
    val builder = tf.currentGraph.nodeBuilder(op, ops.convertNameScopeToName(tf.currentNameScope))
    setAttr(builder)
    val _op = builder.build()
    val outputs = _op.c_op.node().num_outputs()
    Array(outputs) { Output(_op, it) }
  }
}

/**
 * Creates an `Operation` in this graph.
 * This is a low-level interface for creating an [Op]. Most
 * programs will not call this method directly, and instead use the
 * Kotlin op constructors, such as [tf.const], which add ops to
 * the default graph.
 * @see "tensorflow.python.framework.ops.Graph#create_op"
 */
fun createOp(op: String, name: String, inputs: List<Output>, attrs: Map<String, org.tensorflow.framework.AttrValue>): Op =
    tf.nameScope(name) {
      val builder = tf.currentGraph.nodeBuilder(op, ops.convertNameScopeToName(tf.currentNameScope))
      val opDef = tf.currentGraph.getOpDef(op)
      builder.reconstructSequenceInputs(opDef, inputs, attrs)
      attrs.forEach { key, attr -> builder.attr(key, attr) }
      builder.build()
    }

private fun OperationBuilder.reconstructSequenceInputs(
    opDef: OpDef, inputs: List<Output>, attrs: Map<String, org.tensorflow.framework.AttrValue>) {
  var inputLen: Int
  var isSequence: Boolean
  var i = 0
  opDef.inputArgList.forEach { inputArg ->
    when {
      inputArg.numberAttr.isNotEmpty() -> {
        inputLen = attrs[inputArg.numberAttr!!]!!.i.toInt()
        isSequence = true
      }
      inputArg.typeListAttr.isNotEmpty() -> {
        inputLen = attrs[inputArg.typeListAttr!!]!!.list.typeList.size
        isSequence = true
      }
      else -> {
        inputLen = 1
        isSequence = false
      }
    }
    if (isSequence)
      addInput(inputs.subList(i, i + inputLen))
    else
      addInput(inputs[i])
    i += inputLen
  }
  assert(i == inputs.size)
}

class OperationBuilder(val opType: String, val name: String) {
  
  private val scope = graphConstructionScope.value
  private val graph = scope.graph
  private val c_op_desc: TF_OperationDescription = TF_NewOperation(graph.c_graph, opType, name)
  private val inputFunctions = mutableListOf<() -> Unit>()
  private val inputs = mutableListOf<Output>()
  private val inputLists = mutableListOf<Collection<Output>>()
  private val device: String = ""
  private val attributes = mutableMapOf<String, () -> Unit>()
  /** We add an explicit colocation constraint between
   *  the newly created op and any of its reference-typed inputs.*/
  private val maybeColocateInputs = mutableListOf<Op>()
  
  fun build(): Op {
    inputFunctions.forEach { it() }
    processControlInput()
    processColocate()
    processAttributes()
    val c_op = finishOp()
    val op = Op(graph, c_op)
    op.controlFlowContext = tf.currentControlFlowContext
    op.inputs.forEach { }
    control_flow_post_processing(op)
    return op
  }
  
  fun finishOp(): TF_Operation {
    val status = newStatus()
    val nativeOp = TF_FinishOperation(c_op_desc, status)
    status.check()
    return nativeOp
  }
  
  /**
   *strongly connected component indicates equivalent colocationOps
   */
  private fun processColocate() {
    if (scope.colocationOps.isEmpty() && maybeColocateInputs.isEmpty()) return
    val colcated = hashSetOf<Op>()
    val visited = hashSetOf<Op>()
    val queue = ArrayDeque(scope.colocationOps)
    queue.addAll(maybeColocateInputs)
    while (queue.isNotEmpty()) {
      val op = queue.pop()
      if (op !in visited) {
        visited += op
        val dep = op.colocationOps
        if (dep.isEmpty())
          colcated += op
        else
          dep.forEach { if (it !in visited) queue += dep }
      }
    }
    val opDevice = scope.device
    colcated.asSequence().sortedBy { it.name }.forEach { op ->
      if (opDevice != "" && op.device != "" && opDevice != op.device)
        logger.warn {
          "Tried to colocate '$name' with an op '${op.name}' that has a different device: " +
              "$opDevice vs ${op.device}. Ignoring the colocation property."
        }
      else {
        TF_ColocateWith(c_op_desc, op.c_op)
        SetRequestedDevice(graph.c_graph, op.c_op, opDevice)
      }
    }
    TF_SetDevice(c_op_desc, device)
  }
  
  private fun processAttributes() {
    scope.attributes.forEach { key, value ->
      if (key !in attributes)
        attributes[key] = { attr(key, value) }
    }
    attributes.values.forEach { addAttr ->
      addAttr()
    }
  }
  
  private fun control_flow_post_processing(op: Op) {
    op.inputs.forEach { checkInputFromValidContext(op, it.op!!) }
    op.controlFlowContext?.addOp(op)
  }
  
  private fun processControlInput() {
    val controlDependencies = HashSet(scope.controlDependencies)
    input_ops.forEach {
      pruneControlDependencies(controlDependencies, it)
    }
    controlDependencies.forEach { TF_AddControlInput(c_op_desc, it.c_op) }
  }
  
  private fun addInput(input: TF_Output) {
    TF_AddInput(c_op_desc, input)
  }
  
  private val input_ops = mutableSetOf<Op>()
  
  fun addInput(input: Output, ref: Boolean = false) {
    inputFunctions += {
      addInput(input.asTF_Output())
    }
    inputs += input
    if (ref) maybeColocateInputs += input.op!!
  }
  
  fun addInput(input: List<Output>, ref: Boolean = false) {
    inputFunctions += {
      val inputs = TF_Output(input.size.toLong())
      for ((i, _input) in input.withIndex()) {
        val t = _input
        inputs.position(i.toLong()).oper(t.op!!.c_op).index(t.valueIndex)
            .apply {
              input_ops += t.op
              inputLists += input
            }
      }
      TF_AddInputList(c_op_desc, inputs.position(0L), input.size)
    }
    if (ref) input.forEach { maybeColocateInputs += it.op!! }
    input.forEach { inputs += it }
  }
  
  fun attr(name: String, value: String) {
    attributes[name] = {
      attr(name, value.toByteArray())
    }
  }
  
  fun attr(name: String, values: Array<String>) {
    attributes[name] = {
      val attrValue = AttrValue()
      attrValue.mutable_list().apply {
        values.forEach { value ->
          add_s(value)
        }
      }
      attr(name, attrValue)
    }
  }
  
  fun attr(name: String, value: ByteArray) {
    attributes[name] = {
      TF_SetAttrString(c_op_desc, name, BytePointer(*value), value.size.toLong())
    }
  }
  
  fun attr(name: String, value: Boolean) {
    attributes[name] = {
      TF_SetAttrBool(c_op_desc, name, value.toByte())
    }
  }
  
  fun attr(name: String, value: BooleanArray) {
    attributes[name] = {
      TF_SetAttrBoolList(c_op_desc, name,
                         ByteArray(value.size) { value[it].toByte() }, value.size)
    }
  }
  
  fun attr(name: String, value: Int) {
    attributes[name] = {
      TF_SetAttrInt(c_op_desc, name, value.toLong())
    }
  }
  
  fun attr(name: String, value: IntArray) {
    attributes[name] = {
      TF_SetAttrIntList(c_op_desc, name,
                        LongArray(value.size) { value[it].toLong() }, value.size)
    }
  }
  
  fun attr(name: String, value: Long) {
    attributes[name] = {
      TF_SetAttrInt(c_op_desc, name, value)
    }
  }
  
  fun attr(name: String, value: LongArray) {
    attributes[name] = {
      TF_SetAttrIntList(c_op_desc, name, value, value.size)
    }
  }
  
  fun attr(name: String, value: Array<Long>) {
    attributes[name] = {
      TF_SetAttrIntList(c_op_desc, name, value.toLongArray(), value.size)
    }
  }
  
  fun attr(name: String, value: Float) {
    attributes[name] = {
      TF_SetAttrFloat(c_op_desc, name, value)
    }
  }
  
  fun attr(name: String, value: FloatArray) {
    attributes[name] = {
      TF_SetAttrFloatList(c_op_desc, name, value, value.size)
    }
  }
  
  fun attr(name: String, value: Array<Float>) {
    attributes[name] = {
      TF_SetAttrFloatList(c_op_desc, name, value.toFloatArray(), value.size)
    }
  }
  
  fun attr(name: String, dtype: DataType<*>) {
    attributes[name] = {
      TF_SetAttrType(c_op_desc, name, dtype.cValue)
    }
  }
  
  fun <T : Any> attr(name: String, value: Tensor<T>) {
    attributes[name] = {
      val status = newStatus()
      TF_SetAttrTensor(c_op_desc, name, value.c_tensor, status)
      status.check()
    }
  }
  
  fun attr(name: String, value: Shape) {
    attributes[name] = {
      TF_SetAttrShape(c_op_desc, name, value.asLongArray(), value.rank)
    }
  }
  
  fun attr(name: String, value: NDArray<*>) {
    attributes[name] = {
      val status = newStatus()
      TF_SetAttrTensor(c_op_desc, name, Tensor.fromNDArray(value).c_tensor, status)
      status.check()
    }
  }
  
  fun attr(name: String, value: TensorProto) {
    attributes[name] = {
      val attrValue = AttrValue()
      attrValue.mutable_tensor().CopyFrom(value)
      attr(name, attrValue)
    }
  }
  
  fun attr(name: String, shapes: Array<Shape>) {
    attributes[name] = {
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
  }
  
  fun attr(name: String, func: NameAttrList) {
    attributes[name] = {
      val attrValue = AttrValue()
      attrValue.mutable_func().put(func)
      attr(name, attrValue)
    }
  }
  
  fun attr(name: String, attrValue: AttrValue) {
    attributes[name] = {
      val status = newStatus()
      val buf = attrValue.SerializeAsString()
      TF_SetAttrValueProto(c_op_desc, name, buf, buf.limit(), status)
      status.check()
    }
  }
  
  fun attr(name: String, attrValue: org.tensorflow.framework.AttrValue) {
    attributes[name] = {
      val status = newStatus()
      val buf = BytePointer(*attrValue.toByteArray())
      TF_SetAttrValueProto(c_op_desc, name, buf, buf.limit(), status)
      status.check()
    }
  }
  
  fun attr(name: String, tensor_shape_proto: TensorShapeProto) {
    attributes[name] = {
      val status = newStatus()
      val buf = tensor_shape_proto.SerializeAsString()
      TF_SetAttrTensorShapeProto(c_op_desc, name, buf, buf.limit(), status)
      status.check()
    }
  }
}