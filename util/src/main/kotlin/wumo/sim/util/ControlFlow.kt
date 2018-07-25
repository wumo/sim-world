@file:Suppress("UNCHECKED_CAST", "NOTHING_TO_INLINE")

package wumo.sim.util

import org.apache.commons.lang3.ClassUtils

class SwitchValue<T, P1, R> {
  private val branches = HashMap<T, (P1) -> R>()
  fun case(vararg b: T, block: (P1) -> R) {
    for (t in b)
      branches[t] = block
  }
  
  operator fun invoke(b: T, p1: P1): R {
    val block = branches[b] ?: throw IllegalArgumentException("unsupported $b")
    return block(p1)
  }
}

class SwitchValue2<T, P1, P2, R> {
  private val branches = HashMap<T, tuple2<P1, P2>.() -> R>()
  fun case(vararg b: T, block: tuple2<P1, P2>.() -> R) {
    for (t in b)
      branches[t] = block
  }
  
  operator fun invoke(b: T, p1: P1, p2: P2): R {
    val block = branches[b] ?: throw IllegalArgumentException("unsupported $b")
    return block(tuple2(p1, p2))
  }
}

class SwitchValue3<T, P1, P2, P3, R> {
  private val branches = HashMap<T, tuple3<P1, P2, P3>.() -> R>()
  fun case(vararg b: T, block: tuple3<P1, P2, P3>.() -> R) {
    for (t in b)
      branches[t] = block
  }
  
  operator fun invoke(b: T, p1: P1, p2: P2, p3: P3): R {
    val block = branches[b] ?: throw IllegalArgumentException("unsupported $b")
    return block(tuple3(p1, p2, p3))
  }
}

class SwitchValue4<T, P1, P2, P3, P4, R> {
  private val branches = HashMap<T, tuple4<P1, P2, P3, P4>.() -> R>()
  fun case(vararg b: T, block: tuple4<P1, P2, P3, P4>.() -> R) {
    for (t in b)
      branches[t] = block
  }
  
  operator fun invoke(b: T, p1: P1, p2: P2, p3: P3, p4: P4): R {
    val block = branches[b] ?: throw IllegalArgumentException("unsupported $b")
    return block(tuple4(p1, p2, p3, p4))
  }
}

class SwitchValue5<T, P1, P2, P3, P4, P5, R> {
  private val branches = HashMap<T, tuple5<P1, P2, P3, P4, P5>.() -> R>()
  fun case(vararg b: T, block: tuple5<P1, P2, P3, P4, P5>.() -> R) {
    for (t in b)
      branches[t] = block
  }
  
  operator fun invoke(b: T, p1: P1, p2: P2, p3: P3, p4: P4, p5: P5): R {
    val block = branches[b] ?: throw IllegalArgumentException("unsupported $b")
    return block(tuple5(p1, p2, p3, p4, p5))
  }
}

class SwitchValue6<T, P1, P2, P3, P4, P5, P6, R> {
  private val branches = HashMap<T, tuple6<P1, P2, P3, P4, P5, P6>.() -> R>()
  fun case(vararg b: T, block: tuple6<P1, P2, P3, P4, P5, P6>.() -> R) {
    for (t in b)
      branches[t] = block
  }
  
  operator fun invoke(b: T, p1: P1, p2: P2, p3: P3, p4: P4, p5: P5, p6: P6): R {
    val block = branches[b] ?: throw IllegalArgumentException("unsupported $b")
    return block(tuple6(p1, p2, p3, p4, p5, p6))
  }
}

inline fun put(branches: MutableMap<Class<*>, Any>, c: Class<*>, block: Any) {
  if (ClassUtils.isPrimitiveOrWrapper(c)) {
    val cw = if (ClassUtils.isPrimitiveWrapper(c)) c else ClassUtils.primitiveToWrapper(c)
    val c = ClassUtils.wrapperToPrimitive(cw)
    branches[c] = block
    branches[cw] = block
  } else
    branches[c] = block
}

class SwitchType<R> {
  val branches = HashMap<Class<*>, (Any) -> R>()
  inline fun <reified P1> case(noinline block: (P1) -> R) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  operator fun invoke(b: Any): R {
    val block = branches[b::class.java] ?: throw IllegalArgumentException("unsupported ${b::class.java}")
    return block(b)
  }
}


class SwitchType2<P2, R> {
  val branches = HashMap<Class<*>, tuple2<Any, P2>.() -> R>()
  inline fun <reified P1> case(noinline block: tuple2<P1, P2>.() -> R) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  operator fun invoke(b: Any, p2: P2): R {
    val block = branches[b::class.java] ?: throw IllegalArgumentException("unsupported ${b::class.java}")
    return block(tuple2(b, p2))
  }
}

class SwitchType3<P2, P3, R> {
  val branches = HashMap<Class<*>, tuple3<Any, P2, P3>.() -> R>()
  inline fun <reified P1> case(noinline block: tuple3<P1, P2, P3>.() -> R) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  operator fun invoke(b: Any, p2: P2, p3: P3): R {
    val block = branches[b::class.java] ?: throw IllegalArgumentException("unsupported ${b::class.java}")
    return block(tuple3(b, p2, p3))
  }
}

class SwitchType4<P2, P3, P4, R> {
  val branches = HashMap<Class<*>, tuple4<Any, P2, P3, P4>.() -> R>()
  inline fun <reified P1> case(noinline block: tuple4<P1, P2, P3, P4>.() -> R) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  operator fun invoke(b: Any, p2: P2, p3: P3, p4: P4): R {
    val block = branches[b::class.java] ?: throw IllegalArgumentException("unsupported ${b::class.java}")
    return block(tuple4(b, p2, p3, p4))
  }
}

class SwitchType5<P2, P3, P4, P5, R> {
  val branches = HashMap<Class<*>, tuple5<Any, P2, P3, P4, P5>.() -> R>()
  inline fun <reified P1> case(noinline block: tuple5<P1, P2, P3, P4, P5>.() -> R) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  operator fun invoke(b: Any, p2: P2, p3: P3, p4: P4, p5: P5): R {
    val block = branches[b::class.java] ?: throw IllegalArgumentException("unsupported ${b::class.java}")
    return block(tuple5(b, p2, p3, p4, p5))
  }
}

class SwitchType6<P2, P3, P4, P5, P6, R> {
  val branches = HashMap<Class<*>, tuple6<Any, P2, P3, P4, P5, P6>.() -> R>()
  inline fun <reified P1> case(noinline block: tuple6<P1, P2, P3, P4, P5, P6>.() -> R) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  operator fun invoke(b: Any, p2: P2, p3: P3, p4: P4, p5: P5, p6: P6): R {
    val block = branches[b::class.java] ?: throw IllegalArgumentException("unsupported ${b::class.java}")
    return block(tuple6(b, p2, p3, p4, p5, p6))
  }
}