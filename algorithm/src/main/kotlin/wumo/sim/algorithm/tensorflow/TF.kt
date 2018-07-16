package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import org.tensorflow.framework.GraphDef
import wumo.sim.algorithm.tensorflow.ops.group
import wumo.sim.util.println
import java.util.*

var tf = TF()

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
  val scopes = ArrayDeque<Scope>().apply { addLast(Scope()) }
  inline val ctx
    get() = scopes.last
  
  inline fun <R> init_subsope(name: String, device: String = "", block: Scope.() -> R): R {
    scopes.addLast(scopes.first)//添加初始scope
    try {
      return subscope(name, device, block)
    } finally {
      scopes.removeLast()
    }
  }
  
  inline fun <R> init_scope(block: Scope.() -> R): R {
    scopes.addLast(scopes.first)//添加初始scope
    try {
      return block(ctx)
    } finally {
      scopes.removeLast()
    }
  }
  
  inline fun <R> subscope(name: String, device: String = "", block: Scope.() -> R): R {
    scopes.addLast(ctx.newSubscope(name, device))
    try {
      return block(ctx)
    } finally {
      scopes.removeLast()
    }
  }
  
  inline fun <R> with_device(dev: String, block: Scope.() -> R): R =
      ctx.with_device(dev) { block(ctx) }
  
  inline fun <R> colocate_with(colocate_with: Tensor, block: Scope.() -> R) =
      colocate_with(colocate_with.op) { block(ctx) }
  
  inline fun <R> colocate_with(colocate_with: Operation, block: Scope.() -> R) =
      ctx.colocate_with(colocate_with) { block(ctx) }
  
  inline fun <R> control_dependencies(control_inputs: List<Operation>, block: Scope.() -> R) =
      ctx.control_dependencies(control_inputs) { block(ctx) }
  
  /**
   * @see [Scope.condCtx]
   */
  inline fun <R> condCtx(pred: Tensor, pivot: Tensor, branch: Int, block: Scope.() -> R) =
      ctx.condCtx(pred, pivot, branch) { block(ctx) }
  
  fun debugString() = GraphDef.parseFrom(g.toGraphDef()).toString()
  fun printGraph() {
    tf.debugString().println()
  }
  
  fun session(block: Session.() -> Unit) {
    block(Session(g.c_graph))
  }
  
  fun global_variable_initializer(): Operation {
    return group(global_variables, name = "init")
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