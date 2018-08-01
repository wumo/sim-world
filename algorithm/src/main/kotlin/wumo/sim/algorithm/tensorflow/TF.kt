package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import org.tensorflow.framework.GraphDef
import wumo.sim.algorithm.tensorflow.ops.CondContext
import wumo.sim.algorithm.tensorflow.ops.ControlFlowContext
import wumo.sim.algorithm.tensorflow.ops.group
import wumo.sim.algorithm.tensorflow.scope.NameScope
import wumo.sim.algorithm.tensorflow.scope.VariableScope
import wumo.sim.util.println

var tf = TF()
inline fun <R> defaut(_tf: TF, block: () -> R): R {
  val tmp = tf
  tf = _tf
  try {
    return block()
  } finally {
    tf = tmp
  }
}

const val scopeChar = '$'

class TF {
  companion object {
    init {
      Loader.load(tensorflow::class.java)
      tensorflow.InitMain("trainer", null as IntArray?, null)
    }
  }
  
  val g = Graph(this)
  val trainables = mutableListOf<Variable>()
  val global_variables = mutableListOf<Variable>()
  val train_ops = mutableListOf<Operation>()
  val rootNs = NameScope("$", null)
  val rootVs = VariableScope("$", rootNs)
  var ctxNs = rootNs
  var ctxVs = rootVs
  var control_flow_context: ControlFlowContext? = null
  val attr_scope_map = hashMapOf<String, AttrValue>()
  lateinit var session: Session
  /**
  A context manager that lifts ops out of control-flow scopes and function-building graphs.
  
  There is often a need to lift variable initialization ops out of control-flow
  scopes, function-building graphs, and gradient tapes. Entering an
  `init_scope` is a mechanism for satisfying these desiderata. In particular,
  entering an `init_scope` has three effects:
  
  (1) All control dependencies are cleared the moment the scope is entered;
  this is equivalent to entering the context manager returned from
  `control_dependencies(None)`, which has the side-effect of exiting
  control-flow scopes like `tf.cond` and `tf.while_loop`.
  
  (2) All operations that are created while the scope is active are lifted
  into the lowest context on the `context_stack` that is not building a
  graph function. Here, a context is defined as either a graph or an eager
  context. Every context switch, i.e., every installation of a graph as
  the default graph and every switch into eager mode, is logged in a
  thread-local stack called `context_switches`; the log entry for a
  context switch is popped from the stack when the context is exited.
  Entering an `init_scope` is equivalent to crawling up
  `context_switches`, finding the first context that is not building a
  graph function, and entering it. A caveat is that if graph mode is
  enabled but the default graph stack is empty, then entering an
  `init_scope` will simply install a fresh graph as the default one.
  
  (3) The gradient tape is paused while the scope is active.
   */
  inline fun <R> init_scope(block: () -> R): R {
    val ctx = tf.control_flow_context
    tf.control_flow_context = null
    try {
      ctxNs.control_dependencies {
        return block()
      }
    } finally {
      tf.control_flow_context = ctx
    }
  }
  
  inline fun <R> with(sub: NameScope, block: () -> R): R {
    val parentNs = ctxNs
    ctxNs = sub
    try {
      ctxNs.enter()
      return block()
    } finally {
      ctxNs.exit()
      ctxNs = parentNs
    }
  }
  
  /**
   * 在[ctxNs]下新生成名称为[name]的sub[NameScope]（名称冲突则会重新命名解决冲突），
   * 并将其赋值到[ctxNs]。
   *
   * [block]执行结束后，恢复[ctxNs]为调用[name_scope]之前的[NameScope]
   */
  inline fun <R> name_scope(name: String, device: String = "", block: () -> R): R {
    if (name.startsWith(scopeChar))//already in scope
      return block()
    val parentNs = ctxNs
    val sub = parentNs.new_subscope(name)
    sub.device = device
    ctxNs = sub
    try {
      ctxNs.enter()
      return block()
    } finally {
      ctxNs.exit()
      ctxNs = parentNs
    }
  }
  
  /**
   * - 在[ctxVs]下复用或新生成名称为[name]的sub[VariableScope]（复用和重命名按照[VariableScope]的特殊条件进行），
   * 并将其赋值到[ctxVs]；
   * - 同时在[ctxNs]下新生成名称为[name]的sub[NameScope]（名称冲突则会重新命名解决冲突），
   * 并将其赋值到[ctxNs]。
   *
   * @param reuse 是否重用[Variable]
   */
  inline fun <R> variable_scope(name: String, block: () -> R): R {
    val parentVs = ctxVs
    val parentNs = ctxNs
    val s = parentVs.variable_scope(name, ctxVs.reuse, ctxVs.reenter_increment)
    val subNs = parentNs.new_subscope(name)
    ctxVs = s
    ctxNs = subNs
    try {
      ctxNs.enter()
      ctxVs.enter()
      return block()
    } finally {
      ctxVs.exit()
      ctxNs.exit()
      ctxVs = parentVs
      ctxNs = parentNs
    }
  }
  
  inline fun <R> attr_scope(vararg attrs: Pair<String, AttrValue>, block: () -> R): R {
    val saved_attrs = hashMapOf<String, AttrValue>()
    attrs.forEach { (key, value) ->
      attr_scope_map.compute(key) { k, v ->
        if (v != null) saved_attrs[k] = v
        value
      }
    }
    try {
      return block()
    } finally {
      attrs.forEach { (key, _) ->
        attr_scope_map.compute(key) { k, v ->
          saved_attrs[k]
        }
      }
    }
  }
  
  inline fun <R> on_device(dev: String, block: () -> R): R =
      ctxNs.with_device(dev) { block() }
  
  inline fun <R> colocate_with(colocate_with: Tensor, block: () -> R) =
      colocate_with(colocate_with.op!!) { block() }
  
  inline fun <R> colocate_with(colocate_with: Operation, block: () -> R) =
      ctxNs.colocate_with(colocate_with) { block() }
  
  inline fun <R> control_dependencies(control_inputs: List<Operation>, block: () -> R) =
      ctxNs.control_dependencies(control_inputs) { block() }
  
  inline fun <R> control_dependencies(vararg control_inputs: Operation, block: () -> R) =
      ctxNs.control_dependencies(*control_inputs) { block() }
  
  inline fun <R> control_dependencies(control_inputs: Tensor, block: () -> R) =
      ctxNs.control_dependencies(control_inputs.op!!) { block() }
  
  fun debugString() = GraphDef.parseFrom(g.toGraphDef()).toString()
  fun printGraph() {
    tf.debugString().println()
  }
  
  fun session(block: Session.() -> Unit) {
    session = Session(g.c_graph)
    block(session)
  }
  
  fun global_variable_initializer(): Operation {
    return group(global_variables, name = "init")
  }
  
  inline fun <R> with_device(device: String, block: () -> R) =
      ctxNs.with_device(device, block)
  
  var tmpDev = ""
  inline fun begin_device(device: String) {
    tmpDev = ctxNs.device
  }
  
  inline fun end_device() {
    ctxNs.device = tmpDev
  }
  
  inline fun <R> condContext(pred: Tensor, pivot: Tensor, branch: Int, block: (CondContext) -> R): R {
    val tmp = control_flow_context
    control_flow_context = CondContext(pred, pivot, branch)
    try {
      return block(control_flow_context as CondContext)
    } finally {
      control_flow_context = tmp
    }
  }
}

internal fun throwExceptionIfNotOk(status: TF_Status) {
  val code = TF_GetCode(status)
  if (code == TF_OK) return
  val msg = TF_Message(status).string
  throw when (code) {
    TF_INVALID_ARGUMENT -> IllegalArgumentException(msg)
    TF_UNAUTHENTICATED, TF_PERMISSION_DENIED -> SecurityException(msg)
    TF_RESOURCE_EXHAUSTED, TF_FAILED_PRECONDITION -> IllegalStateException(msg)
    TF_OUT_OF_RANGE -> IndexOutOfBoundsException(msg)
    TF_UNIMPLEMENTED -> UnsupportedOperationException(msg)
    else -> Exception(msg)
  }
}