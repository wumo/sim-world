@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.world.examples.algorithm

import wumo.sim.core.Env
import wumo.sim.util.Rand
import wumo.sim.util.ndarray.NDArray

fun argmax_tie_random(set: IntRange, evaluate: (Int) -> Float): Int {
  val iterator = set.iterator()
  val max_a = mutableListOf(iterator.next())
  var max = evaluate(max_a[0])
  while (iterator.hasNext()) {
    val tmp = iterator.next()
    val p = evaluate(tmp)
    if (p > max) {
      max = p
      max_a.apply {
        clear()
        add(tmp)
      }
    } else if (p == max)
      max_a.add(tmp)
  }
  return max_a[Rand().nextInt(max_a.size)]
}

inline fun HashMap<Int, Float>.scale(s: Float) {
  for (key in keys)
    compute(key) { _, v ->
      v!! * s
    }
}

inline fun HashMap<Int, Float>.scaleAdd(s: Float, x: IntArray) {
  for (i in x) {
    compute(i) { _, v ->
      (v ?: 0.0f) + s
    }
  }
}

inline fun HashMap<Int, Float>.innerProduct(x: IntArray): Float {
  var sum = 0.0f
  for (i in x)
    sum += this[i] ?: 0.0f
  return sum
}

inline fun FloatArray.innerProduct(x: IntArray): Float {
  var sum = 0.0f
  for (i in x)
    sum += this[i]
  return sum
}

inline fun FloatArray.scaleAdd(s: Float, z: HashMap<Int, Float>) {
  for ((i, v) in z.entries)
    this[i] += s * v
}

inline fun FloatArray.scaleAdd(s: Float, x: IntArray) {
  for (i in x)
    this[i] += s
}

inline fun Env<NDArray<Float>, *, Int, *, *>.`True Online Sarsa(λ)`(
    Qfunc: LinearTileCodingFunc,
    π: (NDArray<Float>) -> Int,
    λ: Float,
    α: Float,
    episodes: Int,
    maxStep: Int = Int.MAX_VALUE) {
  val γ = 1.0f
  val X = Qfunc.feature
  val w = Qfunc.w
  val z = HashMap<Int, Float>()
  for (episode in 0 until episodes) {
    print("$episode/$episodes")
    var step = 0
    val s = reset()
    var a = π(s)
    var x = X(s, a)
    z.clear()
    var Q_old = 0.0f
    var terminal = false
    var G = 0.0f
    while (true) {
      step++
      if (terminal || step >= maxStep) break
      render()
      val (s_next, reward, done) = step(a)
      G += γ * reward
      terminal = done
      val tmp1 = (1.0f - α * γ * λ * z.innerProduct(x))
      z.scale(γ * λ)
      z.scaleAdd(tmp1, x)
      
      val Q = w.innerProduct(x)
      var δ = reward - Q
      if (!terminal) {
        val a_next = π(s_next)
        val `x'` = X(s_next, a_next)
        val `Q'` = w.innerProduct(`x'`)
        δ += γ * `Q'`
        w.scaleAdd(α * (δ + Q - Q_old), z)
        w.scaleAdd(-α * (Q - Q_old), x)
        Q_old = `Q'`
        x = `x'`
        a = a_next
      } else {
        w.scaleAdd(α * (δ + Q - Q_old), z)
        w.scaleAdd(-α * (Q - Q_old), x)
      }
    }
    println(" Episode reward $G")
  }
}

inline fun Env<NDArray<Float>, *, Int, *, *>.Play(
    π: (NDArray<Float>) -> Int,
    episodes: Int,
    maxStep: Int = Int.MAX_VALUE) {
  val γ = 1.0f
  for (episode in 0 until episodes) {
    print("$episode/$episodes")
    var step = 0
    val s = reset()
    var a = π(s)
    var terminal = false
    var G = 0.0
    while (true) {
      step++
      if (terminal || step >= maxStep) break
      render()
      val (s_next, reward, done) = step(a)
      G += γ * reward
      terminal = done
      if (!terminal)
        a = π(s_next)
    }
    println(" Episode reward $G")
  }
}