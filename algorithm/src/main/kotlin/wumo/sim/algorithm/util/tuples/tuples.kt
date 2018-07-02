package wumo.sim.algorithm.util.tuples

data class tuple2<A, B>(var _1: A, var _2: B) {
  override fun toString(): String = "($_1,$_2)"
}

data class tuple3<A, B, C>(var _1: A, var _2: B, var _3: C) {
  override fun toString(): String {
    return "($_1,$_2,$_3)"
  }
}

data class tuple4<A, B, C, D>(var _1: A, var _2: B, var _3: C, var _4: D) {
  override fun toString(): String {
    return "($_1,$_2,$_3,$_4)"
  }
}

data class tuple5<A, B, C, D, E>(var _1: A, var _2: B, var _3: C, var _4: D, var _5: E) {
  override fun toString(): String {
    return "($_1,$_2,$_3,$_4,$_5)"
  }
}

data class tuple6<A, B, C, D, E, F>(var _1: A, var _2: B, var _3: C, var _4: D, var _5: E, var _6: F) {
  override fun toString(): String {
    return "($_1,$_2,$_3,$_4,$_5,$_6)"
  }
}