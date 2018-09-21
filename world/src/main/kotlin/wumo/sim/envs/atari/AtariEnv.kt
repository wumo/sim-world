package wumo.sim.envs.atari

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

typealias AtariObsType = NDArray<Byte>
typealias AtariEnvType = Env<AtariObsType, Int, AtariEnv>

class AtariEnv(val game: String = "pong",
               val obs_type: ObsType = ram,
               val frameskip: Pair<Int, Int> = 2 to 5,
               repeat_action_probability: Float = 0f)
  : Env<AtariObsType, Int, AtariEnv> {
  
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
  
  override lateinit var rand: Random
  val game_path = Paths.get(game_dir, "$game.bin").toString()
  val ale = ALEInterface()
  val action_set: IntArray
  val screen_width: Int
  val screen_height: Int
  override val action_space: Space<Int>
  override val observation_space: Space<AtariObsType>
  
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

//    val ale_screen_data = ale.screen.array
//    var j = 0
//    for (i in 0 until screen_width * screen_height) {
//      val zrgb = rgb_palette2[ale_screen_data[i.toLong()].toUByte().toInt()].toUInt()
//      array[j++] = ((zrgb shr 16) and 0xff).toByte()
//      array[j++] = ((zrgb shr 8) and 0xff).toByte()
//      array[j++] = ((zrgb shr 0) and 0xff).toByte()
//    }
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

val rgb_palette2 = intArrayOf(
    0x000000, 0, 0x4a4a4a, 0, 0x6f6f6f, 0, 0x8e8e8e, 0,
    0xaaaaaa, 0, 0xc0c0c0, 0, 0xd6d6d6, 0, 0xececec, 0,
    0x484800, 0, 0x69690f, 0, 0x86861d, 0, 0xa2a22a, 0,
    0xbbbb35, 0, 0xd2d240, 0, 0xe8e84a, 0, 0xfcfc54, 0,
    0x7c2c00, 0, 0x904811, 0, 0xa26221, 0, 0xb47a30, 0,
    0xc3903d, 0, 0xd2a44a, 0, 0xdfb755, 0, 0xecc860, 0,
    0x901c00, 0, 0xa33915, 0, 0xb55328, 0, 0xc66c3a, 0,
    0xd5824a, 0, 0xe39759, 0, 0xf0aa67, 0, 0xfcbc74, 0,
    0x940000, 0, 0xa71a1a, 0, 0xb83232, 0, 0xc84848, 0,
    0xd65c5c, 0, 0xe46f6f, 0, 0xf08080, 0, 0xfc9090, 0,
    0x840064, 0, 0x97197a, 0, 0xa8308f, 0, 0xb846a2, 0,
    0xc659b3, 0, 0xd46cc3, 0, 0xe07cd2, 0, 0xec8ce0, 0,
    0x500084, 0, 0x68199a, 0, 0x7d30ad, 0, 0x9246c0, 0,
    0xa459d0, 0, 0xb56ce0, 0, 0xc57cee, 0, 0xd48cfc, 0,
    0x140090, 0, 0x331aa3, 0, 0x4e32b5, 0, 0x6848c6, 0,
    0x7f5cd5, 0, 0x956fe3, 0, 0xa980f0, 0, 0xbc90fc, 0,
    0x000094, 0, 0x181aa7, 0, 0x2d32b8, 0, 0x4248c8, 0,
    0x545cd6, 0, 0x656fe4, 0, 0x7580f0, 0, 0x8490fc, 0,
    0x001c88, 0, 0x183b9d, 0, 0x2d57b0, 0, 0x4272c2, 0,
    0x548ad2, 0, 0x65a0e1, 0, 0x75b5ef, 0, 0x84c8fc, 0,
    0x003064, 0, 0x185080, 0, 0x2d6d98, 0, 0x4288b0, 0,
    0x54a0c5, 0, 0x65b7d9, 0, 0x75cceb, 0, 0x84e0fc, 0,
    0x004030, 0, 0x18624e, 0, 0x2d8169, 0, 0x429e82, 0,
    0x54b899, 0, 0x65d1ae, 0, 0x75e7c2, 0, 0x84fcd4, 0,
    0x004400, 0, 0x1a661a, 0, 0x328432, 0, 0x48a048, 0,
    0x5cba5c, 0, 0x6fd26f, 0, 0x80e880, 0, 0x90fc90, 0,
    0x143c00, 0, 0x355f18, 0, 0x527e2d, 0, 0x6e9c42, 0,
    0x87b754, 0, 0x9ed065, 0, 0xb4e775, 0, 0xc8fc84, 0,
    0x303800, 0, 0x505916, 0, 0x6d762b, 0, 0x88923e, 0,
    0xa0ab4f, 0, 0xb7c25f, 0, 0xccd86e, 0, 0xe0ec7c, 0,
    0x482c00, 0, 0x694d14, 0, 0x866a26, 0, 0xa28638, 0,
    0xbb9f47, 0, 0xd2b656, 0, 0xe8cc63, 0, 0xfce070, 0
)