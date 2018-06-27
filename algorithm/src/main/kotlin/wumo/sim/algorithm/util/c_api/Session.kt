package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.tensorflow.*
import java.lang.Thread
import java.nio.*

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
  
  fun Operation.eval() {
    val t = fetch(this)
    print(t)
  }
  
  private fun print(t: Tensor) {
    val numElements = t.numElements().toInt()
    when (t.dtype) {
      DataType.FLOAT -> {
        val buf = FloatBuffer.allocate(numElements)
        t.writeTo(buf)
        buf.flip()
        while (buf.hasRemaining())
          println(buf.get())
      }
      DataType.DOUBLE -> {
        val buf = DoubleBuffer.allocate(numElements)
        t.writeTo(buf)
        buf.flip()
        while (buf.hasRemaining())
          println(buf.get())
      }
      DataType.INT32 -> {
        val buf = IntBuffer.allocate(numElements)
        t.writeTo(buf)
        buf.flip()
        while (buf.hasRemaining())
          println(buf.get())
      }
      DataType.INT64 -> {
        val buf = LongBuffer.allocate(numElements)
        t.writeTo(buf)
        buf.flip()
        while (buf.hasRemaining())
          println(buf.get())
      }
      else -> throw IllegalStateException("invalid DataType(${t.dtype})")
    }
  }
  
  fun feedAndTarget(inputOp: Operation, tensor: Tensor, target: Operation) {
    val input = inputOp[0]
    val status = TF_NewStatus()
    val inputs = TF_Output(1)
    inputs.oper(input.op.nativeOp)
    inputs.index(input.idx)
    val inputValues = tensor.nativeTensor
    
    TF_SessionRun(nativeSession, null, inputs, inputValues, 1,
        null, null, 0,
        target.nativeOp, 1,
        null, status)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
  }
  
  fun feedAndEval(inputOp: Operation, tensor: Tensor, outputOp: Operation) {
    val input = inputOp[0]
    val status = TF_NewStatus()
    val inputs = TF_Output(1)
    inputs.oper(input.op.nativeOp)
    inputs.index(input.idx)
    val inputValues = tensor.nativeTensor
    
    val output = outputOp[0]
    val outputs = TF_Output(1)
    outputs.oper(output.op.nativeOp)
    outputs.index(output.idx)
    val outputValues = TF_Tensor()
    TF_SessionRun(nativeSession, null, inputs, inputValues, 1,
        outputs, outputValues, 1,
        null, 0,
        null, status)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
    val t = Tensor.fromHandle(outputValues)
    print(t)
  }
  
  fun target(operation: Operation) {
    val status = TF_NewStatus()
    TF_SessionRun(nativeSession, null, null, null, 0,
        null, null, 0,
        operation.nativeOp, 1,
        null, status)
    throwExceptionIfNotOk(status)
    TF_DeleteStatus(status)
  }
  
  fun fetch(operation: Operation) = fetch(operation[0])
  
  fun fetch(operation: String) = fetch(parseOutput(operation))
  
  fun fetch(output: Output): Tensor {
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