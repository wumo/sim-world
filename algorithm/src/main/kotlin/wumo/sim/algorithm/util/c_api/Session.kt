package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.PointerPointer
import org.bytedeco.javacpp.tensorflow.*
import java.lang.Thread

class Session(val g: Graph) : AutoCloseable {
  lateinit var graphRef: Graph.Reference
  lateinit var nativeSession: TF_Session
  val nativeHandleLock = Object()
  private var numActiveRuns = 0
  
  init {
    g.ref().use {
      allocate(it)
      
      graphRef = g.ref()
    }
  }
  
  override fun close() {
    graphRef.close()
    synchronized(nativeHandleLock) {
      while (numActiveRuns > 0) {
        try {
          nativeHandleLock.wait()
        } catch (e: InterruptedException) {
          Thread.currentThread().interrupt()
          return
        }
      }
      delete()
    }
  }
  
  private fun allocate(it: Graph.Reference) {
    val status = TF_NewStatus()
    val opts = TF_NewSessionOptions()
    nativeSession = TF_NewSession(it.nativeHandle(), opts, status)
    TF_DeleteSessionOptions(opts)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
  }
  
  private fun delete() {
    val status = TF_NewStatus()
    TF_CloseSession(nativeSession, status)
    TF_DeleteSession(nativeSession, status)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
  }
  
  fun fetch(operation: String): Tensor {
    val output = parseOutput(operation)
    val status = TF_NewStatus()
    val outputs = TF_Output(1)
    outputs.oper(output.op.nativeOp)
    outputs.index(output.idx)
    val outputValues = TF_Tensor()
    TF_SessionRun(nativeSession, null, null, null, 0,
        outputs, outputValues, 1,
        null, 0,
        null, status)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
    return Tensor.fromHandle(outputValues)
  }
  
  private fun parseOutput(opName: String): Output {
    val colon = opName.lastIndexOf(':')
    if (colon == -1 || colon == opName.length - 1) {
      return Output(operationByName(opName), 0)
    }
    return try {
      val op = opName.substring(0, colon)
      val index = Integer.parseInt(opName.substring(colon + 1))
      Output(operationByName(op), index)
    } catch (e: NumberFormatException) {
      Output(operationByName(opName), 0)
    }
  }
  
  private fun operationByName(opName: String): Operation {
    return g.operation(opName)
  }
}