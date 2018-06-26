package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.tensorflow

class TF_C {
  val g = Graph()
  
  fun session(block: Session.() -> Unit) {
    Session(g).use {
      block(it)
    }
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