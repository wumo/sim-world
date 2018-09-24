package wumo.sim.envs.classic_control

import com.badlogic.gdx.graphics.Color.BLACK
import com.badlogic.gdx.graphics.Color.YELLOW
import wumo.sim.core.Env
import wumo.sim.graphics.Config
import wumo.sim.graphics.Geom
import wumo.sim.graphics.ShapeType.Line_Strip
import wumo.sim.graphics.Viewer
import wumo.sim.spaces.Box
import wumo.sim.spaces.Discrete
import wumo.sim.util.*
import wumo.sim.util.ndarray.NDArray
import wumo.sim.utils.np_random
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

class MountainCar : Env<NDArray<Float>, Float, Int, Int, MountainCar> {
  companion object {
    val min_position = -1.2f
    val max_position = 0.6f
    val max_speed = 0.07f
    val goal_position = 0.5f
  }
  
  val state = NDArray.zeros(2)
  
  override val action_space = Discrete(3)
  override val observation_space = Box(low = NDArray(f(min_position, -max_speed)),
                                       high = NDArray(f(max_position, max_speed)))
  
  override lateinit var rand: Random
  
  init {
    seed()
    reset()
  }
  
  override fun step(a: Int): t4<NDArray<Float>, Float, Boolean, Map<String, Any>> {
    assert(action_space.contains(a)) { "invalid a:$a" }
    var (position, velocity) = state
    velocity += ((a - 1) * 0.001 + cos(3 * position) * (-0.0025)).toFloat()
    velocity = velocity.coerceIn(-max_speed, max_speed)
    position += velocity
    position = position.coerceIn(min_position, max_position)
    if (position == min_position && velocity < 0) velocity = 0.0f
    val done = position >= goal_position
    val reward = -1.0f
    state[0] = position
    state[1] = velocity
    return t4(state.copy(), reward, done, emptyMap())
  }
  
  override fun reset(): NDArray<Float> {
    state[0] = rand.nextDouble(-0.6, -0.4).toFloat()
    state[1] = 0.0f
    return state.copy()
  }
  
  lateinit var viewer: Viewer
  lateinit var car: Geom
  var scale: Float = 1f
  fun height(x: Float) = sin(3 * x) * .45 + .55
  override fun render() {
    if (!::viewer.isInitialized) {
      val screen_width = 600
      val screen_height = 400
      val world_width = max_position - min_position
      scale = screen_width / world_width
      val carwidth = 40
      val carheight = 20
      viewer = Viewer(Config(screen_width, screen_height, isContinousRendering = false))
      viewer += Geom(Line_Strip) {
        color(BLACK)
        for (x in (min_position..max_position) / 100) {
          val y = height(x)
          vertex((x - min_position) * scale, y.toFloat() * scale)
        }
      }
      val clearance = 10f
      car = Geom {
        color(BLACK)
        val l = -carwidth / 2f
        val r = carwidth / 2f
        val t = carheight.toFloat()
        val b = 0f
        rect_filled(l, b, r, b, r, t, l, t)
        color(YELLOW)
        circle(carwidth / 4f, 0f, carheight / 2.5f)
        circle(-carwidth / 4f, 0f, carheight / 2.5f)
      }.attr {
        translation.add(0f, clearance)
      }
      viewer += car
      viewer += Geom {
        val flagx = (goal_position - min_position) * scale
        val flagy1 = (height(goal_position) * scale).toFloat()
        val flagy2 = flagy1 + 50
        color(BLACK)
        rectLine(flagx, flagy1, flagx, flagy2, 1f)
        color(YELLOW)
        triangle_filled(flagx, flagy2, flagx, flagy2 - 10, flagx + 25, flagy2 - 5)
      }
      viewer.startAsync()
    }
    val pos = state[0]
    car.translation.set(((pos - min_position) * scale), (height(pos) * scale).toFloat())
    car.rotation = cos(3 * pos)
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