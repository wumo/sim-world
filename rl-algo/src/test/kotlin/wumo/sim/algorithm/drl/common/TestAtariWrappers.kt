package wumo.sim.algorithm.drl.common

import com.badlogic.gdx.graphics.Pixmap
import org.junit.Test
import wumo.sim.graphics.Config
import wumo.sim.graphics.Image
import wumo.sim.graphics.Viewer
import wumo.sim.util.ndarray.NDArray

class TestAtariWrappers {
  lateinit var viewer: Viewer
  lateinit var img: Image
  val scale = 8f
  
  fun NDArray<Float>.toPixmap(): Pixmap {
    val (screen_height, screen_width, n) = shape
    val stacked_screen_width = screen_width * n
    val img = this
    val pixmap = Pixmap(stacked_screen_width, screen_height, Pixmap.Format.RGB888)
    for (_x in 0 until stacked_screen_width)
      for (y in 0 until screen_height) {
        val x = _x % screen_width
        val i = _x / screen_width
        val gray = (img[y, x, 0 + i] * 255).toInt()
        val color = (gray shl 24) or (gray shl 16) or (gray shl 8) or 0xff
        pixmap.drawPixel(_x, y, color)
      }
    return pixmap
  }
  
  fun render(obs: NDArray<Float>) {
    if (!::viewer.isInitialized) {
      val (screen_height, screen_width, n) = obs.shape
      viewer = Viewer(Config(screen_width * n * scale.toInt(),
                             screen_height * scale.toInt(),
                             isContinousRendering = false))
      img = Image(scale) { obs.toPixmap() }
      viewer += img
      viewer.startAsync()
    } else {
      img.change { obs.toPixmap() }
    }
    viewer.requestRender()
    Thread.sleep(1000)
  }
  
  @Test
  fun testWrapDeepmind() {
    val _env = make_atari("BreakoutNoFrameskip-v4")
    val env = wrap_deepmind(_env, frame_stack = true)
    
    val episode = 10
    var i = 0
    repeat(episode) {
      env.reset()
      var done = false
      var reward = 0.0
      var i = 0
      while (!done) {
        val a = env.action_space.sample()
        val (ob, _reward, _done, _) = env.step(a)
        render(ob)
//        println(i++)
        reward += _reward
        done = _done
      }
      println(reward)
    }
    env.close()
    if (::viewer.isInitialized)
      viewer.close()
  }
}