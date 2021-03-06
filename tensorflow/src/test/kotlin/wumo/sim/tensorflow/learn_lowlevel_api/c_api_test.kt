package wumo.sim.tensorflow.learn_lowlevel_api

import org.bytedeco.javacpp.BoolPointer
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.AddControlInput
import org.junit.Test
import wumo.sim.tensorflow.tf
import wumo.sim.util.Shape
import wumo.sim.util.f

class c_api_test {
  @Test
  fun `update edge`() {
//    val a = tf.variable(1, name = "a")
    
    val A = tf.const(Shape(4, 1), f(1f, 2f, 3f, 4f))
    val B = tf.const(Shape(4, 1), f(1f, 2f, 3f, 4f))
    val C = tf.matMul(A, B, transposeA = true, transposeB = false)
    val pa = BoolPointer(1L)
    val pb = BoolPointer(1)
    tensorflow.GetNodeAttr(C.node().attrs(), BytePointer("transpose_a"), pa)
    tensorflow.GetNodeAttr(C.node().attrs(), BytePointer("transpose_b"), pb)
    println(pa.get())
    println(pb.get())
//    tensorflow.GetNodeAttr(product.node().attrs(), attr_adj_y, *pb)
//    tf.printGraph()
  }
  
  @Test
  fun `add control input`() {
    val a = tf.const(1f, name = "a")
    tf.controlDependencies(a) {
      val b = tf.const(2f, name = "b")
    }
    
    val c = tf.const(3f, name = "c")
    tf.printGraph()
    
    val g = tf.currentGraph.c_graph.graph()
    g.AddControlEdge(a.node(), c.node())
    tf.printGraph()
  }
  
  @Test
  fun `add control input2`() {
    val a = tf.const(1f, name = "a")
//    tf.controlDependencies(a) {
//      val b = tf.const(2f, name = "b")
//    }
    
    val c = tf.const(3f, name = "c")
    tf.printGraph()
    
    AddControlInput(tf.currentGraph.c_graph, c.op.c_op, a.op.c_op)
    tf.printGraph()
  }
}