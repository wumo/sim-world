package wumo.sim.core

interface Space<E> {
  val n: Int
  
  fun sample(): E
  fun contains(x: E): Boolean
}