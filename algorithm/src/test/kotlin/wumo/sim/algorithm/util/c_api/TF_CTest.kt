package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.TF_Version
import org.junit.Before
import org.junit.Test
import java.nio.FloatBuffer

class TF_CTest {
  @Before
  fun setup() {
    Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
    tensorflow.InitMain("trainer", null as IntArray?, null)
  }
  
  @Test
  fun `print version`() {
    println(TF_Version().string)
  }
  
  @Test
  fun `graph def test`() {
    val tf = TF_C()
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
    println(String(s))
  }
  
  
  @Test
  fun `const test`() {
    val tf = TF_C()
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
    val tf = TF_C()
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
  fun `copyTo`() {
    val tf = TF_C()
    Tensor.create(arrayOf(floatArrayOf(3f, 4f, 5f), floatArrayOf(6f, 7f, 8f))).use {
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
      val A_array = Array(2) { FloatArray(3) }
      A.copyTo(A_array)
      for (a in A_array) {
        for (b in a) {
          println(b)
        }
      }
    }
  }
}