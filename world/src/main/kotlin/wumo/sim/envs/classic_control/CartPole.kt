package wumo.sim.envs.classic_control

import com.badlogic.gdx.graphics.Color.BLACK
import wumo.sim.core.Env
import wumo.sim.graphics.Config
import wumo.sim.graphics.Geom
import wumo.sim.graphics.ShapeType.Lines
import wumo.sim.graphics.Viewer
import wumo.sim.spaces.Box
import wumo.sim.spaces.Discrete
import wumo.sim.util.f
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.unaryMinus
import wumo.sim.util.t4
import wumo.sim.util.uniform
import wumo.sim.utils.np_random
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

class CartPole : Env<NDArray<Float>, Float, Int, Int, CartPole> {
  companion object {
    val gravity = 9.8f
    val masscart = 1.0f
    val masspole = 0.1f
    val total_mass = (masspole + masscart)
    val length = 0.5f // actually half the pole's length
    val polemass_length = (masspole * length)
    val force_mag = 10.0f
    val tau = 0.02f  // seconds between state updates
    
    // Angle at which to fail the episode
    val theta_threshold_radians = (12 * 2 * PI / 360).toFloat()
    val x_threshold = 2.4f
  }
  
  // Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
  val high = NDArray(f(
      x_threshold * 2,
      Float.MAX_VALUE,
      theta_threshold_radians * 2,
      Float.MAX_VALUE))
  
  val state = NDArray.zeros(4)
  
  var steps_beyond_done = Double.NaN
  
  override lateinit var rand: Random
  
  init {
    seed()
  }
  
  override val action_space = Discrete(2)
  override val observation_space = Box(-high, high)
  
  override fun step(a: Int): t4<NDArray<Float>, Float, Boolean, Map<String, Any>> {
    assert(action_space.contains(a)) { "invalid a:$a" }
    var (x, x_dot, theta, theta_dot) = state
    
    val force = if (a == 1) force_mag else -force_mag
    val costheta = cos(theta)
    val sintheta = sin(theta)
    val temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    val thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0f / 3.0f - masspole * costheta * costheta / total_mass))
    val xacc = temp - polemass_length * thetaacc * costheta / total_mass
    x += tau * x_dot
    x_dot += tau * xacc
    theta += tau * theta_dot
    theta_dot += tau * thetaacc
    state[0] = x
    state[1] = x_dot
    state[2] = theta
    state[3] = theta_dot
    val done = x < -x_threshold
        || x > x_threshold
        || theta < -theta_threshold_radians
        || theta > theta_threshold_radians
    val reward = when {
      !done -> 1.0f
      steps_beyond_done.isNaN() -> {
        steps_beyond_done = 0.0
        1.0f
      }
      else -> {
        if (steps_beyond_done == 0.0)
          println("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
        steps_beyond_done += 1
        0.0f
      }
    }
    return t4(state.copy(), reward, done, emptyMap())
  }
  
  override fun reset(): NDArray<Float> {
    val s = rand.uniform(-0.05f, 0.05f, 4)
    state.setFrom(s)
//    arrayCopy(s, state, state.size)
    steps_beyond_done = Double.NaN
    return s
  }
  
  lateinit var viewer: Viewer
  lateinit var cart: Geom
  lateinit var pole: Geom
  override fun render() {
    val screen_width = 600
    val screen_height = 400
    
    val world_width: Float = (x_threshold * 2).toFloat()
    val scale: Float = screen_width / world_width
    val carty = 100f // TOP OF CART
    val polewidth = 10f
    val polelen = scale * 1f
    val cartwidth = 50f
    val cartheight = 30f
    
    if (!::viewer.isInitialized) {
      val axleoffset = cartheight / 4f
      viewer = Viewer(Config(screen_width, screen_height, isContinousRendering = false))
      cart = Geom {
        val l = -cartwidth / 2f
        val r = cartwidth / 2f
        val t = cartheight / 2f
        val b = -cartheight / 2f
        color(BLACK)
        rect_filled(l, b, r, b, r, t, l, t)
      }.apply { z = -2f }
      pole = Geom {
        val l = -polewidth / 2f
        val r = polewidth / 2f
        val t = polelen - polewidth / 2f
        val b = -polewidth / 2f
        color(.8f, .6f, .4f, 1f)
        rect_filled(l, b, r, b, r, t, l, t)
      }.apply {
        z = -1f
      }.attr {
        translation.set(0f, axleoffset)
            .add(cart.translation)
      }
      val axle = Geom {
        color(.5f, .5f, .8f, 1f)
        circle(0f, axleoffset, polewidth / 2f)
      }.attr {
        translation.set(cart.translation)
      }
      val track = Geom(Lines) {
        color(BLACK)
        line(0f, carty, screen_width.toFloat(), carty)
      }
      viewer += cart
      viewer += pole
      viewer += axle
      viewer += track
      viewer.startAsync()
    }
    val x = state
    val cartx = x[0] * scale + screen_width / 2
    cart.translation.set(cartx.toFloat(), carty)
    pole.rotation = (-x[2]).toFloat()
    viewer.requestRender()
    Thread.sleep(1000 / 60)
  }
  
  override fun close() {
    if (::viewer.isInitialized)
      viewer.close()
  }
  
  override fun seed(seed: Long?): List<Long> {
    val (rand, seed) = np_random(seed)
    this.rand = rand
    return listOf(seed)
  }
}
