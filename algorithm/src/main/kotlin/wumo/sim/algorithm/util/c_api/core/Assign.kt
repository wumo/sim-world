package wumo.sim.algorithm.util.c_api.core

import wumo.sim.algorithm.util.c_api.Output
import wumo.sim.algorithm.util.c_api.TF_C

fun TF_C.assign(a: Output, b: Output, name: String = "assign") =
    scope(name) {
      g.opBuilder("Assign", contextPath)
          .addInput(a)
          .addInput(b)
          .build()
    }