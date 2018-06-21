package wumo.sim.algorithm.drl

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.graphics.util.ResourceLoader
import wumo.sim.util.math.Rand
import java.nio.FloatBuffer

fun main(args: Array<String>) {
  Loader.load(tensorflow::class.java)
  
  // Platform-specific initialization routine
  tensorflow.InitMain("trainer", null as IntArray?, null)
  val def = GraphDef()
//  val loc = ResourceLoader.getResource("def.pb")
  TF_CHECK_OK(tensorflow.ReadTextProto(Env.Default(), "resources/mlp.pbtxt", def))
  val session = Session(SessionOptions())
  SetDefaultDevice("/gpu:0", def)
  TF_CHECK_OK(session.Create(def))
  val outputs = TensorVector()
  session.Run(StringTensorPairVector(), StringVector(), StringVector("init_all_vars_op"), outputs)
  
  val x = FloatArray(100 * 32) { Rand().nextFloat() }
  val y = FloatArray(100 * 8) { Rand().nextFloat() }
  val xTensor = Tensor.create(x, TensorShape(100, 32))
  val yTensor = Tensor.create(y, TensorShape(100, 8))
  for (i in 0 until 1000) {
    session.Run(StringTensorPairVector(arrayOf("x", "y"), arrayOf(xTensor, yTensor)),
        StringVector("cost"), StringVector(), outputs)
    val buf = outputs[0].createBuffer<FloatBuffer>()
    val cost = buf.get()
    println("cost: $cost")
    session.Run(StringTensorPairVector(arrayOf("x", "y"), arrayOf(xTensor, yTensor)),
        StringVector(), StringVector("train"), outputs)
    outputs.clear()
  }
  session.close()
  def.close()
}