package wumo.sim.core

interface Space<E> {
  fun sample(): E
  fun contains(x: E): Boolean
}