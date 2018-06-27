package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.tensorflow.*
import java.lang.Thread

class Graph : AutoCloseable {
  internal val nativeGraph = TF_NewGraph()
  private val nativeGraphLock = Object()
  private var refcount = 0
  
  override fun close() {
    synchronized(nativeGraphLock) {
      while (refcount > 0) {
        try {
          nativeGraphLock.wait()
        } catch (e: InterruptedException) {
          Thread.currentThread().interrupt()
          return
        }
      }
      TF_DeleteGraph(nativeGraph)
    }
  }
  
  fun opBuilder(type: String, name: String) = OperationBuilder(this, type, name)
  
  fun ref() = Reference()
  
  inner class Reference : AutoCloseable {
    private var active = true
    
    init {
      synchronized(nativeGraphLock) {
        refcount++
      }
    }
    
    override fun close() {
      synchronized(nativeGraphLock) {
        if (!active) return
        active = false
        if (--refcount == 0)
          nativeGraphLock.notifyAll()
      }
    }
    
    fun nativeHandle() = if (active) nativeGraph else null
  }
  
  fun operation(name: String): Operation {
    synchronized(nativeGraphLock) {
      val op = TF_GraphOperationByName(nativeGraph, name)
      return Operation(this, op)
    }
  }
  
  fun toGraphDef(): ByteArray {
    val buf = TF_NewBuffer()
    val status = TF_NewStatus()
    TF_GraphToGraphDef(nativeGraph, buf, status)
    throwExceptionIfNotOk(status)
    val len = buf.length()
    val bytes = ByteArray(len.toInt())
    val d = buf.data()
    d.capacity<Pointer>(len)
    val data = d.asByteBuffer()
    data.get(bytes)
    TF_DeleteStatus(status)
    TF_DeleteBuffer(buf)
    return bytes
  }
}