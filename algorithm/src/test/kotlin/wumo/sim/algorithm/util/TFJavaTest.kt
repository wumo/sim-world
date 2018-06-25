package wumo.sim.algorithm.util

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.junit.Before
import org.junit.Test
import java.nio.FloatBuffer

class TFJavaTest {
  @Before
  fun setup() {
    Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
    tensorflow.InitMain("trainer", null as IntArray?, null)
  }
  
  @Test
  fun `initialize variable`() {
    val tf = TFJava()
    val v = tf.variable(16 x 4, 9f, "W")
    
    tf.writeTextProto("resources/variable-test.pbtxt")
    
    tf.session {
      runner().addTarget("W/assign").run()
      val outputs = runner().fetch("W").run()
      val buf = FloatBuffer.allocate(outputs[0].numElements())
      outputs[0].writeTo(buf)
      buf.flip()
      println(buf.remaining())
      while (buf.hasRemaining())
        println(buf.get())
    }
  }
  
  @Test
  fun `global variable initializer`() {
    val tf = TFJava()
    val v = tf.variable(16 x 4, 9f, "W")
    val init = tf.global_variable_initializer()
    tf.writeTextProto("resources/variable-test2.pbtxt")
    
    tf.session {
      runner().addTarget("W/assign").run()
      val outputs = runner().fetch("W").run()
      val buf = FloatBuffer.allocate(outputs[0].numElements())
      outputs[0].writeTo(buf)
      buf.flip()
      println(buf.remaining())
      while (buf.hasRemaining())
        println(buf.get())
    }
  }
}