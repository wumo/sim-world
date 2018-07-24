package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import org.tensorflow.framework.GraphDef
import wumo.sim.algorithm.tensorflow.ops.group
import wumo.sim.algorithm.tensorflow.ops.register_math_grad
import wumo.sim.algorithm.tensorflow.scope.NameScope
import wumo.sim.algorithm.tensorflow.scope.VariableScope
import wumo.sim.util.println

var tf = TF()

class TF {
  companion object {
    init {
      Loader.load(tensorflow::class.java)
      tensorflow.InitMain("trainer", null as IntArray?, null)
      register_math_grad()
    }
  }
  
  val g = Graph(this)
  val trainables = mutableListOf<Variable>()
  val global_variables = mutableListOf<Variable>()
  val train_ops = mutableListOf<Operation>()
  val rootNs = NameScope("", null)
  val rootVs = VariableScope("", rootNs)
  var ctxNs = rootNs
  var ctxVs = rootVs
  lateinit var session: Session
  
  inline fun <R> init_scope(block: () -> R): R {
    val tmpNs = ctxNs
    ctxNs = rootNs
    try {
      ctxNs.enter()
      return block()
    } finally {
      ctxNs.exit()
      ctxNs = tmpNs
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