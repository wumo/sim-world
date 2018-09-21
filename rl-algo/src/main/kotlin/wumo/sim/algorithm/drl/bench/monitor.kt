package wumo.sim.algorithm.drl.bench

import wumo.sim.core.Env
import wumo.sim.core.Wrapper

class Monitor<O, A, WrappedENV>(
    env: Env<O, A, WrappedENV>,
    allow_early_resets: Boolean = false
) : Wrapper<O, A, WrappedENV>(env) {

}