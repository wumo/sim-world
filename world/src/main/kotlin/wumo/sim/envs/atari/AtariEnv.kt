package wumo.sim.envs.atari

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.Pixmap
import org.bytedeco.javacpp.IntPointer
import org.bytedeco.javacpp.ale.ALEInterface
import wumo.sim.core.Env
import wumo.sim.core.Space
import wumo.sim.envs.atari.AtariEnv.Companion.ObsType.image
import wumo.sim.envs.atari.AtariEnv.Companion.ObsType.ram
import wumo.sim.graphics.Config
import wumo.sim.graphics.Image
import wumo.sim.graphics.Viewer
import wumo.sim.spaces.Box
import wumo.sim.spaces.Discrete
import wumo.sim.util.*
import wumo.sim.util.ndarray.NDArray
import wumo.sim.utils.hash_seed
import wumo.sim.utils.np_random
import java.nio.file.Paths
import kotlin.experimental.and
import kotlin.math.abs
import kotlin.random.Random

typealias AtariObsType = NDArray<Byte>
typealias AtariEnvType = Env<AtariObsType, Byte, Int, Int, AtariEnv>

class AtariEnv(val game: String = "pong",
               val obs_type: ObsType = ram,
               val frameskip: Pair<Int, Int> = 2 to 5,
               repeat_action_probability: Float = 0f)
  : Env<AtariObsType, Byte, Int, Int, AtariEnv> {
  
  companion object {
    val atari_roms_path = "atari_roms"
    val game_dir: String
    val games: Set<String>
    
    init {
      val (dir, files) = unpackDirToTemp(atari_roms_path)
      game_dir = dir
      games = files.map { it.substringBefore('.') }
          .toSet()
    }
    
    enum class ObsType {
      ram, image
    }
  }
  
  override lateinit var rand: Random
  val game_path = Paths.get(game_dir, "$game.bin").toString()
  val ale = ALEInterface()
  val action_set: IntArray
  val screen_width: Int
  val screen_height: Int
  override val action_space: Space<Int, Int>
  override val observation_space: Space<AtariObsType, Byte>
  
  init {
    require(game in games)
    ale.setFloat("repeat_action_probability", repeat_action_probability)
//    ale.setBool("display_screen", true)
//    ale.setBool("sound", false)
    seed()
    
    val _action_set = ale.minimalActionSet
    action_set = IntArray(_action_set.limit().toInt()) {
      _action_set[it.toLong()]
    }
    action_space = Discrete(action_set.size)
    
    val screen = ale.screen
    screen_width = screen.width().toInt()
    screen_height = screen.height().toInt()
    
    observation_space = when (obs_type) {
      ram -> Box(0.toByte(), 255.toByte(), Shape(128))
      image -> Box(0.toByte(), 255.toByte(), Shape(screen_height, screen_width, 3))
    }
  }
  
  override fun seed(seed: Long?): List<Long> {
    val (rand, seed1) = np_random(seed)
    this.rand = rand
    val seed2 = hash_seed(seed1 + 1) % (1 shl 31)
    ale.setInt("random_seed", abs(seed2).toInt())
    ale.loadROM(game_path)
    return listOf(seed1, seed2)
  }
  
  override fun step(a: Int): t4<AtariObsType, Float, Boolean, Map<String, Any>> {
    var reward = 0f
    val action = action_set[a]
    val num_steps = if (frameskip._1 == frameskip._2) frameskip._1
    else rand.nextInt(frameskip._1, frameskip._2)
    repeat(num_steps) {
      reward += ale.act(action)
    }
    val ob = get_obs()
    
    return t4(ob, reward, ale.game_over(), mapOf("ale.lives" to ale.lives()))
  }
  
  private fun get_obs(): AtariObsType =
      when (obs_type) {
        ram -> getRam()
        image -> getImage()
      }
  
  private fun getRam(): AtariObsType {
    val _ram = ale.ram
    val ram = _ram.array()
    val array = ByteArray(_ram.size().toInt())
    ram.get(array)
    return NDArray(array)
  }
  
  private fun getImage(): AtariObsType {
    val shape = Shape(screen_height, screen_width, 3)
    val array = ByteArray(shape.numElements())
    ale.getScreenRGB(array)
    return NDArray(shape, array)
  }
  
  override fun reset(): AtariObsType {
    ale.reset_game()
    return get_obs()
  }
  
  lateinit var viewer: Viewer
  
  private fun AtariObsType.toPixmap(): Pixmap {
    val img = this
    val pixmap = Pixmap(screen_width, screen_height, Pixmap.Format.RGB888)
    for (x in 0 until screen_width)
      for (y in 0 until screen_height) {
        val r = img[y, x, 0].toUByte().toInt()
        val g = img[y, x, 1].toUByte().toInt()
        val b = img[y, x, 2].toUByte().toInt()
        val color = (r shl 24) or (g shl 16) or (b shl 8) or 0xff
        pixmap.drawPixel(x, y, color)
      }
    return pixmap
  }
  
  lateinit var img: Image
  var scale = 3f
  
  override fun render() {
    val img = getImage()
    if (!::viewer.isInitialized) {
      viewer = Viewer(Config(screen_width * scale.toInt(),
                             screen_height * scale.toInt(),
                             isContinousRendering = false))
      this.img = Image(scale) { img.toPixmap() }
      viewer += this.img
      viewer.startAsync()
    } else {
      this.img.change { img.toPixmap() }
    }
    viewer.requestRender()
    Thread.sleep(1000 / 60)
  }
  
  override fun close() {
    if (::viewer.isInitialized)
      viewer.close()
  }
}