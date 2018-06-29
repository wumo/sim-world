package wumo.sim.algorithm

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import java.nio.charset.Charset

fun main(args: Array<String>) {
  Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
  tensorflow.InitMain("trainer", null as IntArray?, null)
  Graph().use { g ->
    val def=g.toGraphDef()
    val value = "Hello from " + TensorFlow.version()
    
    // Construct the computation graph with a single operation, a constant
    // named "MyConst" with a value "value".
    Tensor.create(value.toByteArray(charset("UTF-8"))).use { t ->
      // The Java API doesn't yet include convenience functions for adding operations.
      g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build()
    }
    
    // Execute the "MyConst" operation in a SessionHelper.
    Session(g).use { s ->
      s.runner().fetch("MyConst").run()[0].use { output -> println(String(output.bytesValue())) }
    }
  }
}