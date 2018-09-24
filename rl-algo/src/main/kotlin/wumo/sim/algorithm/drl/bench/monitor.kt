package wumo.sim.algorithm.drl.bench

import wumo.sim.core.Env
import wumo.sim.core.Wrapper

class Monitor<O, OE : Any, A, AE : Any, WrappedENV>(
    env: Env<O, OE, A, AE, WrappedENV>,
    allow_early_resets: Boolean = false
) : Wrapper<O, OE, A, AE, WrappedENV>(env) {

}