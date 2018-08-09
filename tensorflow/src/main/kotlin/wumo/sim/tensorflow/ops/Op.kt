package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.PointerPointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.tensorflow.core.Graph
import wumo.sim.tensorflow.ops.control_flow_ops.CondContext
import wumo.sim.tensorflow.ops.control_flow_ops.ControlFlowContext
import wumo.sim.tensorflow.ops.ops.COLOCATION_OPS_ATTRIBUTE_NAME
import wumo.sim.tensorflow.ops.ops.COLOCATION_OPS_ATTRIBUTE_PREFIX
import java.util.Collections.emptySet as emptyMutableSet

class OpSpecification(val name: String, val opType: String, val device: String)
typealias DeviceFunction = (OpSpecification) -> String

class Op(val graph: Graph, val c_op: TF_Operation) {
  init {
    graph.cache(c_op)
  }
  
  val name: String by lazy { TF_OperationName(c_op).string }
  val device: String get() = TF_OperationDevice(c_op).string
  val opType: String by lazy { TF_OperationOpType(c_op).string }
  val node: Node by lazy { c_op.node() }
  val attrs: AttrSlice by lazy { node.attrs() }
  
  var controlFlowContext: ControlFlowContext? = null
  fun set_control_flow_context(condContext: CondContext) {
    this.controlFlowContext = condContext
  }
  
  /**
   * Update the input to this findOp at the given index.
   * NOTE: This is for TF internal use only. Please don't use it.
   *
   * @param index the index of the input to update.
   * @param tensor the Output to be used as the input at the given index.
   * @param update_dtype If `False`, the type for this input is not updated.
   */
  internal fun updateInput(index: Int, tensor: Output, update_dtype: Boolean = true) {
    val status = newStatus()
    UpdateEdge(graph.c_graph, tensor.asTF_Output(), asTF_Input(index), status)
    inputs = _loadInputs()
  }
  
  internal fun asTF_Input(input_idx: Int) = run {
    TF_Input().oper(c_op).index(input_idx)
  }
  
  /**
   * Add a new control input to this findOp.
   */
  internal fun addControlInput(op: Op) {
    AddControlInput(graph.c_graph, c_op, op.c_op)
    controlInputs = _loadControlInputs()
  }
  
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (javaClass != other?.javaClass) return false
    
    other as Op
    if (c_op != other.c_op) return false
    return true
  }
  
  override fun hashCode() = c_op.hashCode()
  
  val numOutputs
    get() = TF_OperationNumOutputs(c_op)
  val outputs: List<Output>
    get() = List(numOutputs) {
      Output(Op(graph, c_op), it)
    }
  
  /** Number of inputs to this op (i.e., number of tensors fed as input to this op). */
  var numInputs = 0
    private set
  /** Inputs of this op. Note that these inputs are outputs of other ops and thus have type [[Output]]. */
  var inputs = _loadInputs()
    private set
  
  internal fun _loadInputs() = run {
    numInputs = TF_OperationNumInputs(c_op)
    List(numInputs) {
      val output = TF_OperationInput(asTF_Input(it))
      Output(graph.cache(output.oper()), output.index())
    }
  }
  
  var numControlInputs = 0
    private set
  var controlInputs = _loadControlInputs()
    private set
  
  internal fun _loadControlInputs() = run {
    numControlInputs = TF_OperationNumControlInputs(c_op)
    val control_ops = PointerPointer<TF_Operation>(numControlInputs.toLong())
    TF_OperationGetControlInputs(c_op, control_ops, numControlInputs)
    List(numControlInputs) {
      graph.cache(control_ops.get(TF_Operation::class.java, it.toLong()))
    }
  }
  
  var colocationOps = _loadColocationOps()
    private set
  
  /** Colocation ops for this op (i.e., ops guaranteed to be placed on the same device). */
  fun _loadColocationOps() = run {
    val class_attr = attrStringList(COLOCATION_OPS_ATTRIBUTE_NAME)
    if (class_attr != null) {
      val colocationOps = hashSetOf<Op>()
      for (class_name in class_attr)
        if (class_name.startsWith(COLOCATION_OPS_ATTRIBUTE_PREFIX))
          colocationOps += graph.findOp(class_name.substring(COLOCATION_OPS_ATTRIBUTE_PREFIX.length))!!
      colocationOps.toTypedArray()
    } else
      emptyArray()
  }
  
  fun set_attr(key: String, value: AttrValue) {
    val status = newStatus()
    val _buf = value.SerializeAsString()
    val buf = TF_NewBufferFromString(_buf, _buf.limit())
    SetAttr(graph.c_graph, c_op, key, buf, status)
  }
  
  val output_types: List<Int> by lazy {
    val numOutputs = TF_OperationNumOutputs(c_op)
    
    List(numOutputs) {
      TF_OperationOutputType(TF_Output().oper(c_op).index(it))
    }
  }
  
  fun attrString(name: String): String {
    val value = attrs.Find(name)
    return value.s().string
  }
  
  fun attrStringList(name: String): Array<String>? {
    val value = attrs.Find(name) ?: return null
    val list_value = value.list()
    return Array(list_value.s_size()) {
      list_value.s(it).string
    }
  }
  
  fun attrBool(name: String): Boolean {
    val value = attrs.Find(name)
    return value.b()
  }
  
  fun attrBoolList(name: String): BooleanArray {
    val value = attrs.Find(name)
    val list_value = value.list()
    return BooleanArray(list_value.i_size()) {
      list_value.b(it)
    }
  }
  
  fun attrLong(name: String): Long {
    val value = attrs.Find(name)
    return value.i()
  }
  
  fun attrLongList(name: String): LongArray {
    val value = attrs.Find(name)
    val list_value = value.list()
    return LongArray(list_value.i_size()) {
      list_value.i(it)
    }
  }
  
  fun attrFloat(name: String): Float {
    val value = attrs.Find(name)
    return value.f()
  }
  
  fun attrFloatList(name: String): FloatArray {
    val value = attrs.Find(name)
    val list_value = value.list()
    return FloatArray(list_value.i_size()) {
      list_value.f(it)
    }
  }
  
  val attr: Map<String, Any>
    get() {
      TODO()
    }
  
  override fun toString(): String {
    return """"Op("$name", op=$opType, dev=$device, run=${c_op.node().DebugString().string})"""
  }
}