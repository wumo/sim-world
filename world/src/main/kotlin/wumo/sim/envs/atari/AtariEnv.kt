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
import kotlin.math.abs
import kotlin.random.Random

class AtariEnv(val game: String = "pong",
               val obs_type: ObsType = ram,
               val frameskip: Pair<Int, Int> = 2 to 5,
               repeat_action_probability: Float = 0f) : Env<NDArray<Byte>, Int> {
  
  companion object {
    val atari_roms_path = "atari_roms"
    val game_dir: String = unpackDirToTemp(atari_roms_path)
    val games: Set<String> = listResources(atari_roms_path)
        .map { it.nameWithoutExtension }
        .toSet()
    
    enum class ObsType {
      ram, image
    }
  }
  
  lateinit var rand: Random
  val game_path = Paths.get(game_dir, "$game.bin").toString()
  val ale = ALEInterface()
  val action_set: IntPointer
  val screen_width: Int
  val screen_height: Int
  override val action_space: Space<Int>
  override val observation_space: Space<NDArray<Byte>>
  
  init {
    require(game in games)
    ale.setFloat("repeat_action_probability", repeat_action_probability)
//    ale.setBool("display_screen", true)
//    ale.setBool("sound", false)
    seed()
    
    action_set = ale.minimalActionSet
    action_space = Discrete(action_set.limit().toInt())
    
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
  
  override fun step(a: Int): t4<NDArray<Byte>, Float, Boolean, Map<String, Any>> {
    var reward = 0f
    val action = action_set[a.toLong()]
    val num_steps = if (frameskip._1 == frameskip._2) frameskip._1
    else rand.nextInt(frameskip._1, frameskip._2)
    repeat(num_steps) {
      reward += ale.act(action)
    }
    val ob = get_obs()
    
    return t4(ob, reward, ale.game_over(), mapOf("ale.lives" to ale.lives()))
  }
  
  private fun get_obs(): NDArray<Byte> =
      when (obs_type) {
        ram -> getRam()
        image -> getImage()
      }
  
  private fun getRam(): NDArray<Byte> {
    val _ram = ale.ram
    val ram = _ram.array()
    val array = ByteArray(_ram.size().toInt())
    ram.get(array)
    return NDArray(array)
  }
  
  private fun getImage(): NDArray<Byte> {
    val shape = Shape(screen_height, screen_width, 3)
    val array = ByteArray(shape.numElements())
//    ale.getScreenRGB(array)
    
    val ale_screen_data = ale.screen.array
    var j = 0
    for (i in 0 until screen_width * screen_height) {
      val zrgb = rgb_palette[ale_screen_data[i.toLong()].toUByte().toInt()]
      array[j++] = ((zrgb shr 16) and 0xff).toByte()
      array[j++] = ((zrgb shr 8) and 0xff).toByte()
      array[j++] = ((zrgb shr 0) and 0xff).toByte()
    }
    return NDArray(shape, array)
  }
  
  override fun reset(): NDArray<Byte> {
    ale.reset_game()
    return get_obs()
  }
  
  lateinit var viewer: Viewer
  
  private fun NDArray<Byte>.toPixmap(): Pixmap {
    val img = this
    val pixmap = Pixmap(screen_width, screen_height, Pixmap.Format.RGB888)
    for (x in 0 until screen_width)
      for (y in 0 until screen_height) {
        val r = img[y, x, 0].toUByte().toInt()
        val g = img[y, x, 1].toUByte().toInt()
        val b = img[y, x, 2].toUByte().toInt()
        val color = (r shl 16) or (g shl 8) or b
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

val rgb_palette = intArrayOf(
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff,
    0x000000, 0x000000, 0x2121ff, 0x3a3a3a, 0xf03c79, 0x797979, 0xff50ff, 0x989898,
    0x7fff00, 0xbcbcbc, 0x7fffff, 0xd9d9d9, 0xffff3f, 0xe9e9e9, 0xffffff, 0xffffff
)