package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.Loader.load
import org.bytedeco.javacpp.tensorflow
import org.tensorflow.framework.GraphDef

class TF_C {
  companion object {
    init {
      load(tensorflow::class.java)
      tensorflow.InitMain("trainer", null as IntArray?, null)
    }
  }
  
  val g = Graph()
  val trainables = mutableListOf<Operation>()
  val init_ops = mutableListOf<Operation>()
  
  val scope = NameScope()
  
  inner class NameScope {
    val namesInUse = HashSet<String>()
    val scope = mutableListOf<String>()
    val scopeString = StringBuilder()
    
    fun enter(subScope: String) {
      if (scope.isNotEmpty())
        scopeString.append("/")
      scopeString.append(subScope)
      scope += subScope
    }
    
    fun exit() {
      val last = scope.removeAt(scope.lastIndex)
      val slash = if (scope.isEmpty()) 0 else 1
      scopeString.delete(scopeString.length - last.length - slash, scopeString.length)
    }
    
    val contextPath: String
      get() = scopeString.toString()
    
    val useContextName = ""
    
    inline operator fun <R> invoke(name: String, block: NameScope.() -> R): R {
      val enterSubscope = name.isNotEmpty()
      if (enterSubscope) {
        enter(name)
        var newName = scopeString.toString()
        var i = 1
        while (namesInUse.contains(newName)) {
          exit()
          enter("${name}_$i")
          newName = scopeString.toString()
          i++
        }
        namesInUse += newName
      }
      try {
        return block(this)
      } finally {
        if (enterSubscope) exit()
      }
    }
  }
  
  fun global_variable_initializer(): Operation {
    scope("init") {
      return g.opBuilder("NoOp", contextPath)
          .apply {
            for (init_op in init_ops) {
              addControlInput(init_op)
            }
          }.build()
    }
  }
  
  fun session(block: Session.() -> Unit) {
    Session(g).use {
      block(it)
    }
    g.close()
  }
  
  fun debugString() = GraphDef.parseFrom(g.toGraphDef()).toString()
  fun close() {
    g.close()
  }
}

internal fun throwExceptionIfNotOk(status: tensorflow.TF_Status) {
  val code = tensorflow.TF_GetCode(status)
  if (code == tensorflow.TF_OK) return
  val msg = tensorflow.TF_Message(status).string
  throw when (code) {
    tensorflow.TF_INVALID_ARGUMENT -> IllegalArgumentException(msg)
    tensorflow.TF_UNAUTHENTICATED, tensorflow.TF_PERMISSION_DENIED -> SecurityException(msg)
    tensorflow.TF_RESOURCE_EXHAUSTED, tensorflow.TF_FAILED_PRECONDITION -> IllegalStateException(msg)
    tensorflow.TF_OUT_OF_RANGE -> IndexOutOfBoundsException(msg)
    tensorflow.TF_UNIMPLEMENTED -> UnsupportedOperationException(msg)
    else -> Exception(msg)
  }
}