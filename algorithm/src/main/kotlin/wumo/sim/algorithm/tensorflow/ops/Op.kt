package wumo.sim.algorithm.tensorflow.ops

import org.bytedeco.javacpp.PointerPointer
import org.bytedeco.javacpp.helper.tensorflow.AbstractTF_Status.newStatus
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.Graph
import wumo.sim.algorithm.tensorflow.ops.gradients.iterate
import wumo.sim.algorithm.tensorflow.ops.gradients.toTensor

class Op(val graph: Graph, val c_op: TF_Operation) {
  init {
    graph.opsCache[c_op.address()] = this
  }
  
  val name: String by lazy { TF_OperationName(c_op).string }
  val device: String by lazy { TF_OperationDevice(c_op).string }
  val opType: String by lazy { TF_OperationOpType(c_op).string }
  val node: Node by lazy { c_op.node() }
  val attrs: AttrSlice by lazy { node.attrs() }
  
  var control_flow_context: ControlFlowContext? = null
  fun set_control_flow_context(condContext: CondContext) {
    this.control_flow_context = condContext
  }
  
  /**
   * Update the input to this findOp at the given index.
   * NOTE: This is for TF internal use only. Please don't use it.
   *
   * @param index the index of the input to update.
   * @param tensor the Output to be used as the input at the given index.
   * @param update_dtype If `False`, the type for this input is not updated.
   */
  internal fun update_input(index: Int, tensor: Output, update_dtype: Boolean = true) {
    val g = graph.c_graph
    val status = newStatus()
    UpdateEdge(g, tensor.asTF_Output(), _tf_input(index), status)
    inputs = _inputs()
  }
  
  internal fun _tf_input(input_idx: Int) = run {
    TF_Input().oper(c_op).index(input_idx)
  }
  
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (javaClass != other?.javaClass) return false
    
    other as Op
    if (c_op != other.c_op) return false
    return true
  }
  
  override fun hashCode() = c_op.hashCode()
  
  private var _numInputs = TF_OperationNumInputs(c_op)
  
  val numInputs
    get() = _numInputs
  val numOutputs
    get() = TF_OperationNumOutputs(c_op)
  
  fun _inputs() = run {
    val node = c_op.node()
    val numInputs = node.num_inputs()
    val inputs = MutableList(numInputs) { Output(null, it) }
    for (e in node.in_edges().iterate()) {
      if (e.IsControlEdge()) continue
      inputs[e.dst_input()] = toTensor(e.src(), e.src_output())
    }
    inputs
  }
  
  var inputs: List<Output> = _inputs()
  val outputs: List<Output> by lazy {
    List(numOutputs) {
      Output(Op(graph, c_op), it)
    }
  }
  
  /**
   * Add a new control input to this findOp.
   */
  fun addControlInput(op: Op) {
    val g = graph.c_graph
    AddControlInput(g, c_op, op.c_op)
  }
  
  fun set_attr(key: String, value: AttrValue) {
    val status = newStatus()
    val _buf = value.SerializeAsString()
    val buf = TF_NewBufferFromString(_buf, _buf.limit())
    SetAttr(graph.c_graph, c_op, key, buf, status)
  }
  
  val control_inputs: List<Op>
    get() {
      val numControlOps = TF_OperationNumControlInputs(c_op)
      val control_ops = PointerPointer<TF_Operation>(numControlOps.toLong())
      TF_OperationGetControlInputs(c_op, control_ops, numControlOps)
      return List(numControlOps) {
        Op(graph, control_ops.get(TF_Operation::class.java, it.toLong()))
      }
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
    return """"Op("$name", op=$opType, dev=$device, def=${c_op.node().DebugString().string})"""
  }
}