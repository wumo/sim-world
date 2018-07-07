package wumo.sim.algorithm.tensorflow

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import org.tensorflow.framework.GraphDef
import java.util.*

class TF {
  companion object {
    init {
      Loader.load(tensorflow::class.java)
      tensorflow.InitMain("trainer", null as IntArray?, null)
    }
  }
  
  val g = Graph()
  val trainables = mutableListOf<Tensor>()
  val init_ops = mutableListOf<Operation>()
  val scopes = ArrayDeque<Scope>().apply { addLast(Scope()) }
  inline val ctx
    get() = scopes.last
  
  inline fun <R> subscope(name: String, block: () -> R): R {
    scopes.addLast(ctx.newSubscope(name))
    try {
      return block()
    } finally {
      scopes.removeLast()
    }
  }
  
  fun debugString() = GraphDef.parseFrom(g.toGraphDef()).toString()
  
  fun session(block: Session.() -> Unit) {
    block(Session(g.c_graph))
  }
  
  fun global_variable_initializer(): Operation {
    subscope("init") {
      return g.nodeBuilder("NoOp", ctx.name)
          .apply {
            for (init_op in init_ops) {
              addControlInput(init_op)
            }
          }.build()
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