package wumo.sim.algorithm.drl.deepq

interface Schedule {
  fun value(t: Int): Float {
    TODO("not implemented")
  }
}

class NoneSchedule : Schedule {

}