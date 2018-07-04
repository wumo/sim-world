package wumo.sim.algorithm.util.cpp_api.gradient

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.tensorflow.*
import org.junit.Test
import wumo.sim.algorithm.util.cpp_api.BaseTest
import wumo.sim.algorithm.util.cpp_api.ops.add
import wumo.sim.algorithm.util.cpp_api.ops.const

class SymbolicGradientBuilderTest : BaseTest() {
  @Test
  fun `test edgeset iterate`() {
    val a = tf.const(1, "a")
    val b = tf.const(2, "b")
    val `a+b` = tf.add(a, b, "add")
    val n = `a+b`.node()
    val in_edges = n.in_edges()
    
    val size = in_edges.size()
    val iter = EdgeSetIterator(in_edges.begin())
    val end = EdgeSetIterator(in_edges.end())
    var i = 0
    while (iter.notEquals(end)) {
      val ptrptr = iter.access()
      val e = Edge(ptrptr.get())
      println(e.DebugString().string)
      iter.increment()
      i++
    }
    println(size)
    println(i)
  }
  
  val noGradient = Output(Node(Pointer()), -1)
  @Test
  fun `output reference`() {
    val out = Output(Node(Pointer()), 0)
    println(out.equals(noGradient))
    println(out.index())
    out.put<Output>(noGradient)
    println(out.equals(noGradient))
    println(out.index())
  }
}