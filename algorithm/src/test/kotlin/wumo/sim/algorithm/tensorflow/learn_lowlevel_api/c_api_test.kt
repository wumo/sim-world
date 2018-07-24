package wumo.sim.algorithm.tensorflow.learn_lowlevel_api

import org.bytedeco.javacpp.BoolPointer
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.tensorflow
import org.junit.Test
import wumo.sim.algorithm.tensorflow.Operation
import wumo.sim.algorithm.tensorflow.ops.const
import wumo.sim.algorithm.tensorflow.ops.matmul
import wumo.sim.algorithm.tensorflow.tf
import wumo.sim.util.f
import wumo.sim.util.println
import wumo.sim.util.x

class c_api_test {
  @Test
  fun `update edge`() {
    val A = tf.const(4 x 1, f(1f, 2f, 3f, 4f))
    val B = tf.const(4 x 1, f(1f, 2f, 3f, 4f))
    val C = tf.matmul(A, B, transpose_a = true, transpose_b = false)
    val pa = BoolPointer(1L)
    val pb = BoolPointer(1)
    tensorflow.GetNodeAttr(C.node().attrs(), BytePointer("transpose_a"), pa)
    tensorflow.GetNodeAttr(C.node().attrs(), BytePointer("transpose_b"), pb)
    println(pa.get())
    println(pb.get())
//    tensorflow.GetNodeAttr(product.node().attrs(), attr_adj_y, *pb)
    tf.printGraph()
    
  }
}