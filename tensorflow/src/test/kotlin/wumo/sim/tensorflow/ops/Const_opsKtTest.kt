package wumo.sim.tensorflow.ops

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.tensorflow
import org.junit.Test
import wumo.sim.tensorflow.ops.basic.toProto
import wumo.sim.tensorflow.tf
import wumo.sim.util.Shape
import wumo.sim.util.a
import wumo.sim.util.f
import wumo.sim.util.scalarDimension

class Const_opsKtTest : BaseTest() {
  
  @Test
  fun const1() {
    val p = scalarDimension.toProto()
    val s = p.SerializeAsString()
    val sss = BytePointer()
    val pp = tensorflow.TensorShapeProto()
    pp.ParseFromString("")
    println(pp.dim_size())
    
    val a = tf.const(1f)
    val b = tf.const(2f, name = "b")
    val c = tf.const(Shape(2, 2), 6L, name = "c")
    val d = tf.const(Shape(2, 2), "hello", name = "d")
    printGraph()
  }
  
  @Test
  fun const2() {
    val c = tf.const(Shape(2, 2, 2), 6f)
    val d = tf.const(Shape(2, 2), f(1f, 2f, 3f, 4f), name = "d")
    val e = tf.const(Shape(2, 2), a("hello", "tensorflow", "and", "you"), name = "e")
    val f = tf.const(a("hello", "tensorflow", "and", "you"), name = "f")
    printGraph()
    tf.session {
      c.eval()
      d.eval()
      e.eval()
      f.eval()
    }
  }
  
  @Test
  fun const3() {
    val d = tf.const(Shape(2, 3), f(1f, 2f, 3f, 4f, 5f, 6f), name = "d")
    tf.session {
      val _d = eval<Float>(d)
      println(_d)
    }
  }
}