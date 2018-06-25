package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.tensorflow.TF_DeleteGraph
import org.bytedeco.javacpp.tensorflow.TF_NewGraph

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
  }
}