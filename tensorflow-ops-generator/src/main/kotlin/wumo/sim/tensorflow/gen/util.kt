package wumo.sim.tensorflow.gen

import wumo.sim.util.sb

fun String.toCamelCase(ignoreLeadingUnderscore: Boolean = true): String = sb {
  val cs = this@toCamelCase.toCharArray()
  for ((i, c) in cs.withIndex())
    if (c == '_' && i + 1 < cs.size) {
      if (i == 0) {
        if (ignoreLeadingUnderscore)
          cs[i + 1] = cs[i + 1].toLowerCase()
        else
          +c
      } else
        cs[i + 1] = cs[i + 1].toUpperCase()
    } else
      +c
}