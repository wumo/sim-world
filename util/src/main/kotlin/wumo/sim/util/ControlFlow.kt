@file:Suppress("UNCHECKED_CAST", "NOTHING_TO_INLINE")

package wumo.sim.util

import org.apache.commons.lang3.ClassUtils

typealias Fun1<P1, R> = (P1) -> R
typealias Fun2<P1, P2, R> = t2<P1, P2>.() -> R
typealias Fun3<P1, P2, P3, R> = t3<P1, P2, P3>.() -> R
typealias Fun4<P1, P2, P3, P4, R> = t4<P1, P2, P3, P4>.() -> R
typealias Fun5<P1, P2, P3, P4, P5, R> = t5<P1, P2, P3, P4, P5>.() -> R
typealias Fun6<P1, P2, P3, P4, P5, P6, R> = t6<P1, P2, P3, P4, P5, P6>.() -> R

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
    return block(t2(p1, p2))
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
    return block(t3(p1, p2, p3))
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
    return block(t4(p1, p2, p3, p4))
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
    return block(t5(p1, p2, p3, p4, p5))
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
    return block(t6(p1, p2, p3, p4, p5, p6))
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

class SwitchOnClass<R> {
  val branches = HashMap<Class<*>, Fun1<Any, R>>()
  var elseBranch: (Any) -> R = { throw IllegalArgumentException("unsupported ${it::class.java}") }
  inline fun <reified P1> case(noinline block: Fun1<P1, R>) {
    put(branches as MutableMap<Class<*>, Any>, P1::class.java, block)
  }
  
  inline fun caseElse(noinline block: Fun1<Any, R>) {
    elseBranch = block
  }
  
  operator fun invoke(cls: Class<*>): R {
    val block = branches[cls] ?: return elseBranch(cls)
    return block(cls)
  }
  
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
    val params = t2(b, p2)
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
  var elseBranch: t3<Any, P2, P3>.() -> R = { throw IllegalArgumentException("unsupported ${this._1::class.java}") }
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
    val params = t3(b, p2, p3)
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
  var elseBranch: t4<Any, P2, P3, P4>.() -> R = { throw IllegalArgumentException("unsupported ${this._1::class.java}") }
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
    val params = t4(b, p2, p3, p4)
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
  var elseBranch: t5<Any, P2, P3, P4, P5>.() -> R = { throw IllegalArgumentException("unsupported ${this._1::class.java}") }
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
    val params = t5(b, p2, p3, p4, p5)
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
  var elseBranch: t6<Any, P2, P3, P4, P5, P6>.() -> R = { throw IllegalArgumentException("unsupported ${this._1::class.java}") }
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
    val params = t6(b, p2, p3, p4, p5, p6)
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

inline fun <T> Class<T>.primitive() =
    if (ClassUtils.isPrimitiveWrapper(this))
      ClassUtils.wrapperToPrimitive(this)
    else this

inline fun <T> Class<T>.wrapper() =
    if (ClassUtils.isPrimitiveWrapper(this))
      this
    else ClassUtils.primitiveToWrapper(this)

inline fun putReturnType(branches: MutableMap<Pair<Class<*>, Class<*>>, Any>, c: Pair<Class<*>, Class<*>>, block: Any) {
  val (paramType, returnType) = c
  val primitiveP = paramType.primitive()
  val wrapperP = paramType.wrapper()
  val primitiveR = returnType.primitive()
  val wrapperR = returnType.wrapper()
  
  branches[primitiveP to primitiveR] = block
  branches[primitiveP to wrapperR] = block
  branches[wrapperP to primitiveR] = block
  branches[wrapperP to primitiveR] = block
}

class SwitchReturnType {
  val branches = HashMap<Pair<Class<*>, Class<*>>, Fun1<Any, Any>>()
  val subtypeBranches = HashMap<Pair<Class<*>, Class<*>>, Fun1<Any, Any>>()
  var elseBranch: (Any) -> Any = { throw IllegalArgumentException("unsupported ${it::class.java}") }
  inline fun <reified P1, reified R> case(noinline block: Fun1<P1, R>) {
    putReturnType(branches as MutableMap<Pair<Class<*>, Class<*>>, Any>,
                  P1::class.java to R::class.java, block)
  }
  
  inline fun caseElse(noinline block: Fun1<Any, Any>) {
    elseBranch = block
  }
  
  inline fun <reified P1, reified R> caseIs(noinline block: Fun1<P1, Any>) {
    putReturnType(subtypeBranches as MutableMap<Pair<Class<*>, Class<*>>, Any>,
                  P1::class.java to R::class.java, block)
  }
  
  inline operator fun <reified R> invoke(b: Any): R {
    val cls = b::class.java to R::class.java
    
    val block = branches[cls]
    if (block == null) {
      for ((supertype, fn) in subtypeBranches) {
        if (supertype.first.isAssignableFrom(cls.first)
            && supertype.second.isAssignableFrom(cls.second))
          return fn(b) as R
      }
      return elseBranch(b) as R
    }
    return block(b) as R
  }
}

interface ReturnType<P, R>

class SwitchReturnTypeClass {
  val branches = HashMap<Pair<Class<*>, Class<*>>, ReturnType<Any, Any>>()
  inline fun <reified P1, reified R> case(block: ReturnType<P1, R>) {
    putReturnType(branches as MutableMap<Pair<Class<*>, Class<*>>, Any>,
                  P1::class.java to R::class.java, block)
  }
  
  inline operator fun <reified R, reified T> invoke(b: T): ReturnType<T, R> {
    val cls = T::class.java to R::class.java
    
    val block = branches[cls] ?: throw IllegalArgumentException("unsupported $cls")
    return block as ReturnType<T, R>
  }
}
