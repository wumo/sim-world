package wumo.sim.algorithm.drl.deepq

import kotlin.math.min

/**
 * Linear interpolation between initial_p and final_p over
 * schedule_timesteps. After this many timesteps pass final_p is
 * returned.
 * @param schedule_timesteps Number of timesteps for which to linearly anneal initial_p to final_p
 * @param initial_p initial output value
 * @param final_p final output value
 */
class LinearSchedule(val schedule_timesteps: Int,
                     val final_p: Float,
                     val initial_p: Float = 1f) : Schedule {
  override fun value(t: Int): Float {
    val fraction = min(t.toFloat() / schedule_timesteps, 1.0f)
    return initial_p + fraction * (final_p - initial_p)
  }
}