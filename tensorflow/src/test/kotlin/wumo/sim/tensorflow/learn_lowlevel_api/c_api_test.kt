package wumo.sim.tensorflow.learn_lowlevel_api

import org.bytedeco.javacpp.BoolPointer
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.AddControlInput
import org.junit.Test
import wumo.sim.tensorflow.ops.const
import wumo.sim.tensorflow.ops.gen.matMul
import wumo.sim.tensorflow.ops.variable
import wumo.sim.tensorflow.tf
import wumo.sim.util.f
import wumo.sim.util.x

class c_api_test {
  @Test
  fun `update edge`() {
    val a = tf.variable(1, name = "a")
    
    val A = tf.const(4 x 1, f(1f, 2f, 3f, 4f))
    val B = tf.const(4 x 1, f(1f, 2f, 3f, 4f))
    val C = tf.matMul(A, B, transpose_a = true, transpose_b = false)
    val pa = BoolPointer(1L)
    val pb = BoolPointer(1)
    tensorflow.GetNodeAttr(C.node().attrs(), BytePointer("transpose_a"), pa)
    tensorflow.GetNodeAttr(C.node().attrs(), BytePointer("transpose_b"), pb)
    println(pa.get())
    println(pb.get())
//    tensorflow.GetNodeAttr(product.node().attrs(), attr_adj_y, *pb)
    tf.printGraph()
  }
  
  @Test
  fun `add control input`() {
    val a = tf.const(1f, name = "a")
    tf.control_dependencies(a) {
      val b = tf.const(2f, name = "b")
    }
    
    val c = tf.const(3f, name = "c")
    tf.printGraph()
    
    val g = tf.g.c_graph.graph()
    g.AddControlEdge(a.node(), c.node())
    tf.printGraph()
  }
  
  @Test
  fun `add control input2`() {
    val a = tf.const(1f, name = "a")
    tf.control_dependencies(a) {
      val b = tf.const(2f, name = "b")
    }
    
    val c = tf.const(3f, name = "c")
    tf.printGraph()
    
    AddControlInput(tf.g.c_graph, c.op!!.c_op, a.op!!.c_op)
    tf.printGraph()
  }
}