package wumo.sim.tensorflow

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.core.Graph
import wumo.sim.algorithm.tensorflow.ops.Op
import wumo.sim.algorithm.tensorflow.ops.Output
import wumo.sim.algorithm.tensorflow.ops.control_flow_ops.control_flow_ops.checkInputFromValidContext
import wumo.sim.algorithm.tensorflow.ops.ops
import wumo.sim.algorithm.tensorflow.ops.ops.logger
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.toByte
import wumo.sim.util.warn
import java.util.*
import java.util.Collections.emptySet as emptyMutableSet

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
    Output(_op, 0)
  }
}

fun TF.buildOpTensors(op: String, name: String, setAttr: OperationBuilder.() -> Unit = {}) = run {
  name_scope(name) {
    val builder = g.nodeBuilder(op, ctxNs.scopeName.fullName)
    setAttr(builder)
    val _op = builder.build()
    val outputs = _op.c_op.node().num_outputs()
    Array(outputs) { Output(_op, it) }
  }
}

class OperationBuilder(val graph: Graph, val opType: String, val name: String) {
  private val c_op_desc: TF_OperationDescription = TF_NewOperation(graph.c_graph, opType, name)
  private val inputFunctions = mutableListOf<() -> Unit>()
  private val inputs = mutableListOf<Output>()
  private val inputLists = mutableListOf<Array<Output>>()
  private val device: String = ""
  private val attributes = mutableMapOf<String, () -> Unit>()
  
  fun build(): Op {
    processControlInput()
    processColocate()
    processAttributes()
    val c_op = finishOp()
    val op = Op(graph, c_op)
    op.controlFlowContext = ops.currentControlFlowContext
    op.inputs.forEach { }
    control_flow_post_processing(op)
    return op
  }
  
  fun finishOp(): TF_Operation {
    val status = newStatus()
    val nativeOp = TF_FinishOperation(c_op_desc, status)
    throwExceptionIfNotOk(status)
    return nativeOp
  }
  
  /**
   *strongly connected component indicates equivalent colocationOps
   */
  private fun processColocate() {
    if (tf.colocate_with.isEmpty()) return
    val colcated = hashSetOf<Op>()
    val visited = hashSetOf<Op>()
    val queue = ArrayDeque(tf.colocate_with)
    while (queue.isNotEmpty()) {
      val op = queue.pop()
      if (op !in visited) {
        visited += op
        val dep = op.colocationOps
        if (dep.isEmpty())
          colcated += dep
        else
          dep.forEach { if (it !in visited) queue += dep }
      }
    }
    val opDevice = ops.device
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
    ops.attr_scope_map.forEach { key, value ->
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
    val controlDependencies = HashSet(tf.control_ops)
    input_ops.forEach {
      pruneControlDependencies(controlDependencies, it)
    }
    controlDependencies.forEach { TF_AddControlInput(c_op_desc, it.c_op) }
  }
  
  /** Prunes control dependencies from the provided set, given that the op for which these control dependencies are
   * specified uses `op` as direct or indirect (through other ops) input or control input. This eliminates redundant
   * control dependencies due to transitive dependencies (e.g., if `a` depends on `b` and `c`, and `b` depends on
   * `c`, then the dependency of `a` on `c` is pruned).
   *
   * @param  controlDependencies  Current set of control dependencies for the op that is being built.
   * @param  op           Op that is a direct or indirect (through other ops) input or control input, for the op that
   *                      is being built.
   * @param  processedOps Already processed ops (provided for efficiency purposes so that we do not go through them
   *                      a second time).
   */
  private fun pruneControlDependencies(
      controlDependencies: MutableSet<Op>,
      op: Op,
      processedOps: MutableSet<Op> = emptyMutableSet(),
      maxDepth: Int = 10
  ) {
    if (maxDepth > 0 && op !in processedOps) {
      // Prune op that is already used as input to the dependant op
      controlDependencies -= op
      processedOps += op
      // Prune transitive control dependencies
      op.inputs.forEach { pruneControlDependencies(controlDependencies, it.op!!, processedOps, maxDepth - 1) }
      op.controlInputs.forEach { pruneControlDependencies(controlDependencies, it, processedOps, maxDepth - 1) }
    }
  }
  
  private fun addInput(input: TF_Output) {
    TF_AddInput(c_op_desc, input)
  }
  
  private val input_ops = mutableSetOf<Op>()
  private fun checkRef(input: Output, ref: Boolean) =
      if (ref) input.asRef() else input.value()
  
  fun addInput(input: Output, ref: Boolean = false) {
    inputFunctions += {
      addInput(checkRef(input, ref).asTF_Output())
    }
    inputs += input
  }
  
  fun addInput(input: Collection<Output>, ref: Boolean = false) {
    inputFunctions += {
      val inputs = TF_Output(input.size.toLong())
      for ((i, _input) in input.withIndex()) {
        val t = checkRef(_input, ref)
        inputs.position(i.toLong()).oper(t.op!!.c_op).index(t.value_index)
      }
      TF_AddInputList(c_op_desc, inputs.position(0L), input.size)
    }
    input.forEach { inputs += it }
  }
  
  fun addInput(input: Array<Output>, ref: Boolean = false) {
    inputFunctions += {
      val inputs = TF_Output(input.size.toLong())
      for ((i, _input) in input.withIndex()) {
        val t = checkRef(_input, ref)
        inputs.position(i.toLong()).oper(t.op!!.c_op).index(t.value_index)
            .apply {
              input_ops += t.op
              inputLists += input
            }
      }
      TF_AddInputList(c_op_desc, inputs.position(0L), input.size)
    }
    input.forEach { inputs += it }
  }
  
  fun attr(name: String, value: String) = run {
    attributes[name] = {
      attr(name, value.toByteArray())
    }
  }
  
  fun attr(name: String, values: Array<String>) = run {
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
  
  fun attr(name: String, value: ByteArray) = run {
    attributes[name] = {
      TF_SetAttrString(c_op_desc, name, BytePointer(*value), value.size.toLong())
    }
  }
  
  fun attr(name: String, value: Boolean) = run {
    attributes[name] = {
      TF_SetAttrBool(c_op_desc, name, value.toByte())
    }
  }
  
  fun attr(name: String, value: BooleanArray) = run {
    attributes[name] = {
      TF_SetAttrBoolList(c_op_desc, name,
                         ByteArray(value.size) { value[it].toByte() }, value.size)
    }
  }
  
  fun attr(name: String, value: Int) = run {
    attributes[name] = {
      TF_SetAttrInt(c_op_desc, name, value.toLong())
    }
  }
  
  fun attr(name: String, value: IntArray) = run {
    attributes[name] = {
      TF_SetAttrIntList(c_op_desc, name,
                        LongArray(value.size) { value[it].toLong() }, value.size)
    }
  }
  
  fun attr(name: String, value: Long) = run {
    attributes[name] = {
      TF_SetAttrInt(c_op_desc, name, value)
    }
  }
  
  fun attr(name: String, value: LongArray) = run {
    attributes[name] = {
      TF_SetAttrIntList(c_op_desc, name, value, value.size)
    }
  }
  
  fun attr(name: String, value: Array<Long>) = run {
    attributes[name] = {
      TF_SetAttrIntList(c_op_desc, name, value.toLongArray(), value.size)
    }
  }
  
  fun attr(name: String, value: Float) = run {
    attributes[name] = {
      TF_SetAttrFloat(c_op_desc, name, value)
    }
  }
  
  fun attr(name: String, value: FloatArray) = run {
    attributes[name] = {
      TF_SetAttrFloatList(c_op_desc, name, value, value.size)
    }
  }
  
  fun attr(name: String, value: Array<Float>) = run {
    attributes[name] = {
      TF_SetAttrFloatList(c_op_desc, name, value.toFloatArray(), value.size)
    }
  }
  
  fun attrType(name: String, dtype: Int) = run {
    attributes[name] = {
      TF_SetAttrType(c_op_desc, name, dtype)
    }
  }
  
  fun <T : Any> attr(name: String, value: Tensor<T>) = run {
    attributes[name] = {
      val status = newStatus()
      TF_SetAttrTensor(c_op_desc, name, value.c_tensor, status)
      throwExceptionIfNotOk(status)
    }
  }
  
  fun attr(name: String, value: Shape) = run {
    attributes[name] = {
      TF_SetAttrShape(c_op_desc, name, value.asLongArray(), value.rank)
    }
  }
  
  fun attr(name: String, value: NDArray<*>) = run {
    attributes[name] = {
      val status = newStatus()
      TF_SetAttrTensor(c_op_desc, name, Tensor.fromNDArray(value).c_tensor, status)
      throwExceptionIfNotOk(status)
    }
  }
  
  fun attr(name: String, value: TensorProto) = run {
    attributes[name] = {
      val attrValue = AttrValue()
      attrValue.mutable_tensor().CopyFrom(value)
      attr(name, attrValue)
    }
  }
  
  fun attr(name: String, shapes: Array<Shape>) = run {
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
  
  fun attr(name: String, func: NameAttrList) = run {
    attributes[name] = {
      val attrValue = AttrValue()
      attrValue.mutable_func().put(func)
      attr(name, attrValue)
    }
  }
  
  fun attr(name: String, attrValue: AttrValue) = run {
    attributes[name] = {
      val status = newStatus()
      val buf = attrValue.SerializeAsString()
      TF_SetAttrValueProto(c_op_desc, name, buf, buf.limit(), status)
      throwExceptionIfNotOk(status)
    }
  }
  
  fun attr(name: String, tensor_shape_proto: TensorShapeProto) = run {
    attributes[name] = {
      val status = newStatus()
      val buf = tensor_shape_proto.SerializeAsString()
      TF_SetAttrTensorShapeProto(c_op_desc, name, buf, buf.limit(), status)
      throwExceptionIfNotOk(status)
    }
  }
}