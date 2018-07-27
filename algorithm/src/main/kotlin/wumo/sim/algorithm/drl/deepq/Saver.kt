package wumo.sim.algorithm.drl.deepq

import com.badlogic.gdx.math.Rectangle.tmp
import wumo.sim.algorithm.tensorflow.*
import wumo.sim.algorithm.tensorflow.ops.assign
import wumo.sim.algorithm.tensorflow.ops.const
import wumo.sim.algorithm.tensorflow.ops.group
import wumo.sim.util.ndarray.NDArray
import java.io.BufferedOutputStream

class Model {

}

fun saveModel(act_graph_def: ByteArray, act_vars: List<Pair<String, NDArray<Any>>>): ByteArray {
  val _tf = TF()
  _tf.g.import(act_graph_def, "save")
  defaut(_tf) {
    val init_ops = arrayListOf<Operation>()
    for ((v, value) in act_vars)
      init_ops += tf.assign(Tensor(tf.g.operation("save/$v"), 0), tf.const(value.copy())).op!!
    tf.group(init_ops, name = "init")
    return tf.g.toGraphDef()
  }
}
