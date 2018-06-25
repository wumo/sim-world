package wumo.sim.algorithm.test

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.tensorflow.*
import java.nio.FloatBuffer

fun main(args: Array<String>) {
  Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
  tensorflow.InitMain("trainer", null as IntArray?, null)
  
  val graph = Graph()
  var opB = graph.opBuilder("Placeholder", "A")
  opB.setAttr("dtype", DataType.FLOAT)
  opB.setAttr("shape", Shape.make(2, 2))
  val A = opB.build().output<Float>(0)
  
  opB = graph.opBuilder("Const", "b")
  val t = Tensor.create(arrayOf(floatArrayOf(3f, 5f)))
  opB.setAttr("dtype", DataType.FLOAT)
  opB.setAttr("value", t)
  val b = opB.build().output<Float>(0)
  t.close()
  
  graph.opBuilder("MatMul", "v").apply {
    addInput(A)
    addInput(b)
    setAttr("transpose_b", true)
    build()
  }
  val s = graph.toGraphDef()
  println(String(s))
  val def = tensorflow.GraphDef()
  val success = def.ParseFromString(BytePointer(*s))
  println(success)
  if (success) {
    tensorflow.TF_CHECK_OK(tensorflow.WriteTextProto(tensorflow.Env.Default(), "resources/custom-q-network.pbtxt", def))
  }
  def.close()
  val session = Session(graph)
  val output = session.runner().feed("A", Tensor.create(arrayOf(floatArrayOf(3f, 2f), floatArrayOf(-1f, 0f))))
      .fetch("v").run()
  
  for (i in 0 until output.size) {
    val a = output[i]
    println(a)
    val buf = FloatBuffer.allocate(a.numElements())
    a.writeTo(buf)
    buf.flip()
    while (buf.hasRemaining())
      println(buf.get())
  }
  session.close()
  graph.close()
}