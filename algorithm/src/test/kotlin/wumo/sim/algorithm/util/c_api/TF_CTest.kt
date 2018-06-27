package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.TF_Version
import org.junit.Assert
import org.junit.Before
import org.junit.Test
import org.tensorflow.framework.GraphDef
import wumo.sim.algorithm.util.x
import java.nio.FloatBuffer

class TF_CTest {
  lateinit var tf: TF_C
  @Before
  fun setup() {
    Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
    tensorflow.InitMain("trainer", null as IntArray?, null)
    tf = TF_C()
  }
  
  @Test
  fun `print version`() {
    println(TF_Version().string)
  }
  
  @Test
  fun `attrValueProto test`() {
    val attrValue = tensorflow.AttrValue()
    attrValue.mutable_tensor().apply {
      set_dtype(tensorflow.DT_FLOAT)
      mutable_tensor_shape().apply {
        add_dim().set_size(16)
        add_dim().set_size(4)
      }
      add_float_val(9f)
    }
    attrValue.use {
      tf.g.opBuilder("Const", "A")
          .setAttr("dtype", DataType.FLOAT)
          .setAttr("value", it)
          .build()
    }
    tf.session {
      val A = fetch("A")
      val buf = FloatBuffer.allocate(64)
      A.writeTo(buf)
      buf.flip()
      println(buf.remaining())
      while (buf.hasRemaining())
        println(buf.get())
      println(buf.remaining())
    }
  }
  
  @Test
  fun `const def test`() {
    val A = tf.const(16 x 4, 9f, "A")
    
    tf.session {
      A.eval()
    }
  }
  
  @Test
  fun `placeholder test`() {
    val A = tf.placeholder(2 x 2 x 2, "A")
    val B = tf.variable(2 x 2 x 2, DataType.FLOAT, A, "B")
    val init = tf.global_variable_initializer()
    tf.session {
      feedAndTarget(A, Tensor.create(arrayOf(
          arrayOf(floatArrayOf(1f, 2f), floatArrayOf(3f, 4f)),
          arrayOf(floatArrayOf(5f, 6f), floatArrayOf(7f, 8f)))),
          init)
      B.eval()
    }
  }
  
  @Test
  fun `variable def test`() {
    val A = tf.variable(16 x 4, 9f, "A")
    val init = tf.global_variable_initializer()
    println(tf.debugString())
    tf.session {
      target(init)
      A.eval()
    }
  }
  
  @Test
  fun `graph def test`() {
    Tensor.create(floatArrayOf(3f, 4f, 5f)).use {
      tf.g.opBuilder("Const", "A")
          .setAttr("dtype", DataType.FLOAT)
          .setAttr("value", it)
          .build()
    }
    Tensor.create(floatArrayOf(3f, 4f, 5f, 6f, 7f)).use {
      tf.g.opBuilder("Const", "B")
          .setAttr("dtype", DataType.FLOAT)
          .setAttr("value", it)
          .build()
    }
    val s = tf.g.toGraphDef()
    val def = GraphDef.parseFrom(s)
    println(def)
    println(String(s))
  }
  
  
  @Test
  fun `const test`() {
    Tensor.create(9.3f).use {
      tf.g.opBuilder("Const", "A")
          .setAttr("dtype", DataType.FLOAT)
          .setAttr("value", it)
          .build()
    }
    tf.session {
      val A = fetch("A")
      println(A.floatValue())
    }
  }
  
  @Test
  fun `const array test`() {
    Tensor.create(floatArrayOf(3f, 4f, 5f)).use {
      tf.g.opBuilder("Const", "A")
          .setAttr("dtype", DataType.FLOAT)
          .setAttr("value", it)
          .build()
    }
    Tensor.create(floatArrayOf(3f, 4f, 5f, 6f, 7f)).use {
      tf.g.opBuilder("Const", "B")
          .setAttr("dtype", DataType.FLOAT)
          .setAttr("value", it)
          .build()
    }
    tf.session {
      val A = fetch("A")
      val B = fetch("B")
      val buf = FloatBuffer.allocate(3)
      A.writeTo(buf)
      buf.flip()
      println(buf.remaining())
      while (buf.hasRemaining())
        println(buf.get())
      println(buf.remaining())
    }
  }
  
  @Test
  fun `write 3d tensor`() {
    var i = 0f
    val tensor = Array(2) {
      Array(2) {
        FloatArray(3) {
          i++
        }
      }
    }
    
    Tensor.create(tensor).use {
      tf.g.opBuilder("Const", "A")
          .setAttr("dtype", DataType.FLOAT)
          .setAttr("value", it)
          .build()
    }
    tf.session {
      val A = fetch("A")
      val buf = FloatBuffer.allocate(12)
      A.writeTo(buf)
      buf.flip()
      println(buf.remaining())
      i = 0f
      while (buf.hasRemaining()) {
        val acutal = buf.get()
        println(acutal)
        Assert.assertEquals(i++, acutal)
      }
      println(buf.remaining())
    }
  }
  
  @Test
  fun `copyTo 3d tensor`() {
    var i = 0f
    val tensor = Array(2) {
      Array(2) {
        FloatArray(3) {
          i++
        }
      }
    }
    
    Tensor.create(tensor).use {
      tf.g.opBuilder("Const", "A")
          .setAttr("dtype", DataType.FLOAT)
          .setAttr("value", it)
          .build()
    }
    tf.session {
      val A = fetch("A")
      
      val A_array = Array(2) { Array(2) { FloatArray(3) } }
      A.copyTo(A_array)
      i = 0f
      for (a in A_array)
        for (b in a)
          for (c in b)
            Assert.assertEquals(i++, c)
    }
  }
}