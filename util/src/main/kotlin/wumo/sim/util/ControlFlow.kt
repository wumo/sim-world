@file:Suppress("UNCHECKED_CAST", "NOTHING_TO_INLINE")

package wumo.sim.util

import org.apache.commons.lang3.ClassUtils

typealias Fun1<P1, R> = (P1) -> R
typealias Fun2<P1, P2, R> = tuple2<P1, P2>.() -> R
typealias Fun3<P1, P2, P3, R> = tuple3<P1, P2, P3>.() -> R
typealias Fun4<P1, P2, P3, P4, R> = tuple4<P1, P2, P3, P4>.() -> R
typealias Fun5<P1, P2, P3, P4, P5, R> = tuple5<P1, P2, P3, P4, P5>.() -> R
typealias Fun6<P1, P2, P3, P4, P5, P6, R> = tuple6<P1, P2, P3, P4, P5, P6>.() -> R

class SwitchValue<T, P1, R> {
  private val branches = HashMap<T, Fun1<P1, R>>()
  fun case(vararg b: T, block: Fun1<P1, R>) {
    for (t in b)
      branches[t] = block
  }
  
  operator fun invoke(b: T, p1: P1): R {
    val block = branches[b] ?: throw IllegalArgumentException("unsupported $b")
    return block(p1)
  }
}

class SwitchValue2<T, P1, P2, R> {
  private val branches = HashMap<T, Fun2<P1, P2, R>>()
  fun case(vararg b: T, block: Fun2<P1, P2, R>) {
    for (t in b)
      branches[t] = block
  }
  
  operator fun invoke(b: T, p1: P1, p2: P2): R {
    val block = branches[b] ?: throw IllegalArgumentException("unsupported $b")
    return block(tuple2(p1, p2))
  }
}

class SwitchValue3<T, P1, P2, P3, R> {
  private val branches = HashMap<T, Fun3<P1, P2, P3, R>>()
  fun case(vararg b: T, block: Fun3<P1, P2, P3, R>) {
    for (t in b)
      branches[t] = block
  }
  
  operator fun invoke(b: T, p1: P1, p2: P2, p3: P3): R {
    val block = branches[b] ?: throw IllegalArgumentException("unsupported $b")
    return block(tuple3(p1, p2, p3))
  }
}

class SwitchValue4<T, P1, P2, P3, P4, R> {
  private val branches = HashMap<T, Fun4<P1, P2, P3, P4, R>>()
  fun case(vararg b: T, block: Fun4<P1, P2, P3, P4, R>) {
    for (t in b)
      branches[t] = block
  }
  
  operator fun invoke(b: T, p1: P1, p2: P2, p3: P3, p4: P4): R {
    val block = branches[b] ?: throw IllegalArgumentException("unsupported $b")
    return block(tuple4(p1, p2, p3, p4))
  }
}

class SwitchValue5<T, P1, P2, P3, P4, P5, R> {
  private val branches = HashMap<T, Fun5<P1, P2, P3, P4, P5, R>>()
  fun case(vararg b: T, block: Fun5<P1, P2, P3, P4, P5, R>) {
    for (t in b)
      branches[t] = block
  }
  
  operator fun invoke(b: T, p1: P1, p2: P2, p3: P3, p4: P4, p5: P5): R {
    val block = branches[b] ?: throw IllegalArgumentException("unsupported $b")
    return block(tuple5(p1, p2, p3, p4, p5))
  }
}

class SwitchValue6<T, P1, P2, P3, P4, P5, P6, R> {
  private val branches = HashMap<T, Fun6<P1, P2, P3, P4, P5, P6, R>>()
  fun case(vararg b: T, block: Fun6<P1, P2, P3, P4, P5, P6, R>) {
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
  val branches = HashMap<Class<*>, Fun1<Any, R>>()
  val subtypeBranches = HashMap<Class<*>, Fun1<Any, R>>()
  var elseBranch: (Any) -> R = { throw IllegalArgumentException("unsupported ${it::class.java}") }
  inline fun <reified P1> case(noinline block: Fun1<P1, R>) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  inline fun caseElse(noinline block: Fun1<Any, R>) {
    elseBranch = block
  }
  
  inline fun <reified P1> caseIs(noinline block: Fun1<P1, R>) {
    put(subtypeBranches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  operator fun invoke(b: Any): R {
    val cls = b::class.java
    val block = branches[cls]
    if (block == null) {
      for ((supertype, fn) in subtypeBranches) {
        if (supertype.isAssignableFrom(cls))
          return fn(b)
      }
      return elseBranch(b)
    }
    return block(b)
  }
  
}

class SwitchType2<P2, R> {
  val branches = HashMap<Class<*>, Fun2<Any, P2, R>>()
  val subtypeBranches = HashMap<Class<*>, Fun2<Any, P2, R>>()
  var elseBranch: Fun2<Any, P2, R> = { throw IllegalArgumentException("unsupported ${this._1::class.java}") }
  inline fun <reified P1> case(noinline block: Fun2<P1, P2, R>) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  inline fun caseElse(noinline block: Fun2<Any, P2, R>) {
    elseBranch = block
  }
  
  inline fun <reified P1> caseIs(noinline block: Fun2<P1, P2, R>) {
    put(subtypeBranches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  operator fun invoke(b: Any, p2: P2): R {
    val params = tuple2(b, p2)
    val cls = b::class.java
    val block = branches[cls]
    if (block == null) {
      for ((supertype, fn) in subtypeBranches) {
        if (supertype.isAssignableFrom(cls))
          return fn(params)
      }
      return elseBranch(params)
    }
    return block(params)
  }
}

class SwitchType3<P2, P3, R> {
  val branches = HashMap<Class<*>, Fun3<Any, P2, P3, R>>()
  val subtypeBranches = HashMap<Class<*>, Fun3<Any, P2, P3, R>>()
  var elseBranch: tuple3<Any, P2, P3>.() -> R = { throw IllegalArgumentException("unsupported ${this._1::class.java}") }
  inline fun <reified P1> case(noinline block: Fun3<P1, P2, P3, R>) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  inline fun caseElse(noinline block: Fun3<Any, P2, P3, R>) {
    elseBranch = block
  }
  
  inline fun <reified P1> caseIs(noinline block: Fun3<P1, P2, P3, R>) {
    put(subtypeBranches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  operator fun invoke(b: Any, p2: P2, p3: P3): R {
    val params = tuple3(b, p2, p3)
    val cls = b::class.java
    val block = branches[cls]
    if (block == null) {
      for ((supertype, fn) in subtypeBranches) {
        if (supertype.isAssignableFrom(cls))
          return fn(params)
      }
      return elseBranch(params)
    }
    return block(params)
  }
}

class SwitchType4<P2, P3, P4, R> {
  val branches = HashMap<Class<*>, Fun4<Any, P2, P3, P4, R>>()
  val subtypeBranches = HashMap<Class<*>, Fun4<Any, P2, P3, P4, R>>()
  var elseBranch: tuple4<Any, P2, P3, P4>.() -> R = { throw IllegalArgumentException("unsupported ${this._1::class.java}") }
  inline fun <reified P1> case(noinline block: Fun4<P1, P2, P3, P4, R>) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  inline fun caseElse(noinline block: Fun4<Any, P2, P3, P4, R>) {
    elseBranch = block
  }
  
  inline fun <reified P1> caseIs(noinline block: Fun4<P1, P2, P3, P4, R>) {
    put(subtypeBranches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  operator fun invoke(b: Any, p2: P2, p3: P3, p4: P4): R {
    val params = tuple4(b, p2, p3, p4)
    val cls = b::class.java
    val block = branches[cls]
    if (block == null) {
      for ((supertype, fn) in subtypeBranches) {
        if (supertype.isAssignableFrom(cls))
          return fn(params)
      }
      return elseBranch(params)
    }
    return block(params)
  }
}

class SwitchType5<P2, P3, P4, P5, R> {
  val branches = HashMap<Class<*>, Fun5<Any, P2, P3, P4, P5, R>>()
  val subtypeBranches = HashMap<Class<*>, Fun5<Any, P2, P3, P4, P5, R>>()
  var elseBranch: tuple5<Any, P2, P3, P4, P5>.() -> R = { throw IllegalArgumentException("unsupported ${this._1::class.java}") }
  inline fun <reified P1> case(noinline block: Fun5<P1, P2, P3, P4, P5, R>) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  inline fun caseElse(noinline block: Fun5<Any, P2, P3, P4, P5, R>) {
    elseBranch = block
  }
  
  inline fun <reified P1> caseIs(noinline block: Fun5<P1, P2, P3, P4, P5, R>) {
    put(subtypeBranches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  operator fun invoke(b: Any, p2: P2, p3: P3, p4: P4, p5: P5): R {
    val params = tuple5(b, p2, p3, p4, p5)
    val cls = b::class.java
    val block = branches[cls]
    if (block == null) {
      for ((supertype, fn) in subtypeBranches) {
        if (supertype.isAssignableFrom(cls))
          return fn(params)
      }
      return elseBranch(params)
    }
    return block(params)
  }
}

class SwitchType6<P2, P3, P4, P5, P6, R> {
  val branches = HashMap<Class<*>, Fun6<Any, P2, P3, P4, P5, P6, R>>()
  val subtypeBranches = HashMap<Class<*>, Fun6<Any, P2, P3, P4, P5, P6, R>>()
  var elseBranch: tuple6<Any, P2, P3, P4, P5, P6>.() -> R = { throw IllegalArgumentException("unsupported ${this._1::class.java}") }
  inline fun <reified P1> case(noinline block: Fun6<P1, P2, P3, P4, P5, P6, R>) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  inline fun caseElse(noinline block: Fun6<Any, P2, P3, P4, P5, P6, R>) {
    elseBranch = block
  }
  
  inline fun <reified P1> caseIs(noinline block: Fun6<P1, P2, P3, P4, P5, P6, R>) {
    put(subtypeBranches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  operator fun invoke(b: Any, p2: P2, p3: P3, p4: P4, p5: P5, p6: P6): R {
    val params = tuple6(b, p2, p3, p4, p5, p6)
    val cls = b::class.java
    val block = branches[cls]
    if (block == null) {
      for ((supertype, fn) in subtypeBranches) {
        if (supertype.isAssignableFrom(cls))
          return fn(params)
      }
      return elseBranch(params)
    }
    return block(params)
  }
}
