package wumo.sim.algorithm.util.cpp_api.layers

import org.bytedeco.javacpp.tensorflow.*

open class Layer(val trainable: Boolean = true,
                 val activity_reqularizer: Any? = null,
                 dtype: Int = -1,
                 val name: String = "", scope: Scope? = null) {
  var statefule = false
  var built = false
  
}