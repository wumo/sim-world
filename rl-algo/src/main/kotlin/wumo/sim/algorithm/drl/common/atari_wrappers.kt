package wumo.sim.algorithm.drl.common

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_core.Mat.AUTO_STEP
import org.bytedeco.javacpp.opencv_imgproc.*
import wumo.sim.ale.PLAYER_A_FIRE
import wumo.sim.ale.PLAYER_A_NOOP
import wumo.sim.buf
import wumo.sim.core.Env
import wumo.sim.core.ObservationWrapper
import wumo.sim.core.RewardWrapper
import wumo.sim.core.Wrapper
import wumo.sim.envs.atari.AtariEnv
import wumo.sim.envs.atari.AtariEnvType
import wumo.sim.envs.atari.AtariObsType
import wumo.sim.envs.envs
import wumo.sim.spaces.Box
import wumo.sim.util.*
import wumo.sim.util.ndarray.*
import wumo.sim.util.ndarray.types.NDByte
import wumo.sim.util.ndarray.types.NDFloat
import java.util.*
import kotlin.math.sign
import kotlin.random.Random

fun make_atari(env_id: String): AtariEnvType {
  require("NoFrameskip" in env_id)
  var env = envs.Atari(env_id)
  env = NoopResetEnv(env, noop_max = 30)
  env = MaxAndSkipEnv(env, skip = 4)
  return env
}

fun wrap_atari_dqn(env: AtariEnvType) = run {
  var _env = env
  _env = EpisodicLifeEnv(_env)
  if (PLAYER_A_FIRE in _env.unwrapped.action_set)
    _env = FireResetEnv(_env)
  _env = WarpFrame(_env)
  _env = ClipRewardEnv(_env)
  _env = FrameStack(_env, 4)
  env
}

fun wrap_deepmind(env: AtariEnvType,
                  episode_life: Boolean = true,
                  clip_rewards: Boolean = true,
                  frame_stack: Boolean = false,
                  scale: Boolean = false): Env<NDArray<Float>, Float, Int, Int, Any> {
  var _env = env
  if (episode_life)
    _env = EpisodicLifeEnv(_env)
  if (PLAYER_A_FIRE in _env.unwrapped.action_set)
    _env = FireResetEnv(_env)
  _env = WarpFrame(_env)
  var env: Env<NDArray<Float>, Float, Int, Int, Any> = FloatFrame(_env)
  if (scale)
    env = ScaledFloatFrame(env)
  if (clip_rewards)
    env = ClipRewardEnv(env)
  if (frame_stack)
    env = FrameStack(env, 4)
  return env
}

class NoopResetEnv(env: AtariEnvType,
                   val noop_max: Int = 30)
  : Wrapper<AtariObsType, Byte, Int, Int, AtariEnv>(env) {
  
  val noop_action: Int = 0
  var override_num_noops: Int? = null
  
  init {
    require(env.unwrapped.action_set[0] == PLAYER_A_NOOP)//NOOP
  }
  
  override fun reset(): AtariObsType {
    native {
      super.reset()
      val noops = override_num_noops ?: rand.nextInt(1, noop_max + 1)
      require(noops > 0)
      lateinit var obs: AtariObsType
      repeat(noops) {
        val (_obs, _, done) = env.step(noop_action)
        obs = _obs
        if (done)
          obs = env.reset()
      }
      obs.ref()
      return obs
    }
  }
}

class MaxAndSkipEnv(env: AtariEnvType, val skip: Int = 4)
  : Wrapper<AtariObsType, Byte, Int, Int, AtariEnv>(env) {
  
  val obs_buffer = mutableListOf(NDArray(env.observation_space.shape, 0.toByte()),
                                 NDArray(env.observation_space.shape, 0.toByte()))
  
  override fun step(a: Int): t4<AtariObsType, Float, Boolean, Map<String, Any>> {
    native {
      var total_reward = 0f
      var done = false
      lateinit var info: Map<String, Any>
      for (i in 0 until skip) {
        val (obs, reward, _done, _info) = env.step(a)
        done = _done
        info = _info
        if (i == skip - 2) {
          obs_buffer[0].unref()
          obs_buffer[0] = obs.ref()
        }
        if (i == skip - 1) {
          obs_buffer[1].unref()
          obs_buffer[1] = obs.ref()
        }
        total_reward += reward
        if (done) break
      }
      val a = obs_buffer[0]
      val b = obs_buffer[1]
      val size = a.shape.numElements().toLong()
      val c = BytePointer(size)
      buf.maxOf(a.raw.ptr, b.raw.ptr, c, size)
      val max_frame = NDArray(a.shape, BytePointerBuf(c, NDByte))
      max_frame.ref()
      return t4(max_frame, total_reward, done, info)
    }
  }
}

class EpisodicLifeEnv(env: AtariEnvType)
  : Wrapper<AtariObsType, Byte, Int, Int, AtariEnv>(env) {
  
  var lives = 0
  var was_real_done = true
  
  override fun step(a: Int): t4<AtariObsType, Float, Boolean, Map<String, Any>> {
    val result = env.step(a)
    var done = result._3
    was_real_done = done
    val lives = env.unwrapped.lives()
    if (lives < this.lives && lives > 0)
      done = true
    this.lives = lives
    result._3 = done
    return result
  }
  
  override fun reset(): AtariObsType {
    val obs = if (was_real_done)
      env.reset()
    else
      env.step(0)._1
    lives = env.unwrapped.lives()
    return obs
  }
}

class FireResetEnv(env: AtariEnvType)
  : Wrapper<AtariObsType, Byte, Int, Int, AtariEnv>(env) {
  
  init {
    require(env.unwrapped.action_set[1] == PLAYER_A_FIRE)
    require(env.unwrapped.action_set.size >= 3)
  }
  
  override fun reset(): NDArray<Byte> {
    native {
      env.reset()
      val (_, _, done, _) = env.step(1)
      if (done)
        env.reset()
      val (obs, _, _done, _) = env.step(2)
      if (_done)
        env.reset()
      obs.ref()
      return obs
    }
  }
}

fun AtariObsType.toMat(): Mat {
  val channels = shape[2]
  return Mat(shape[0], shape[1], CV_8UC(channels), raw.ptr, AUTO_STEP.toLong())
}

fun Mat.toNDArray(): AtariObsType {
  require(depth() == CV_8U) { "Only supported CV_8U" }
  val channels = channels()
  val data = data()
  val shape = Shape(rows(), cols(), channels)
  data.capacity(shape.numElements().toLong())
  return NDArray(shape,
                 BytePointerBuf(data, NDByte))
}

class WarpFrame(env: AtariEnvType) :
    ObservationWrapper<AtariObsType, Byte, Int, Int, AtariEnv>(env) {
  
  val width = 84
  val height = 84
  
  override val observation_space =
      Box(0.toByte(), 255.toByte(), Shape(height, width, 1), NDByte)
  
  override fun observation(frame: AtariObsType): AtariObsType {
    native {
      val src = frame.toMat()
      val dst = Mat()
      cvtColor(src, dst, COLOR_RGB2GRAY)
      val dst2 = Mat()
      resize(dst, dst2, Size(width, height), 0.0, 0.0, INTER_AREA)
      frame.unref()
      dst2.ref()
      return dst2.toNDArray()
    }
  }
}

class FloatFrame<WrappedEnv>(val env: AtariEnvType)
  : Env<NDArray<Float>, Float, Int, Int, WrappedEnv> {
  
  override var rand: Random = env.rand
  override val action_space = env.action_space
  override val observation_space = run {
    val obspace = env.observation_space as Box<Byte>
    Box(obspace.low.cast(NDFloat), obspace.high.cast(NDFloat))
  }
  
  override fun step(a: Int): t4<NDArray<Float>, Float, Boolean, Map<String, Any>> {
    native {
      val (_obs, reward, done, info) = env.step(a)
      
      val obs = _obs.cast(NDFloat)
      obs.ref()
      return t4(obs, reward, done, info)
    }
  }
  
  override fun reset(): NDArray<Float> {
    native {
      val obs = env.reset()
      return obs.cast(NDFloat).ref()
    }
  }
  
  override fun render() = env.render()
  
  override fun close() = env.close()
  
  override fun seed(seed: Long?) = env.seed(seed)
}

class ScaledFloatFrame<WrappedEnv>(
    env: Env<NDArray<Float>, Float, Int, Int, WrappedEnv>)
  : ObservationWrapper<NDArray<Float>, Float, Int, Int, WrappedEnv>(env) {
  
  override val observation_space = Box(0f, 1f, env.observation_space.shape, NDFloat)
  
  override fun observation(frame: NDArray<Float>): NDArray<Float> {
    frame /= 255f
    return frame
  }
  
}

class ClipRewardEnv<OE : Any, WrappedEnv>(
    env: Env<NDArray<OE>, OE, Int, Int, WrappedEnv>)
  : RewardWrapper<NDArray<OE>, OE, Int, Int, WrappedEnv>(env) {
  
  override fun reward(frame: Float): Float {
    return frame.sign
  }
}

class FixedSizeDeque<E>(val capacity: Int,
                        inline val deallocator: (E) -> Unit)
  : ArrayDeque<E>(capacity + 1) {
  
  override fun addFirst(e: E) {
    if (size == capacity)
      deallocator(removeLast())
    super.addFirst(e)
  }
  
  override fun addLast(e: E) {
    if (size == capacity)
      deallocator(removeFirst())
    super.addLast(e)
  }
}

class FrameStack<OE, WrappedEnv>(
    env: Env<NDArray<OE>, OE, Int, Int, WrappedEnv>, val k: Int)
  : Wrapper<NDArray<OE>, OE, Int, Int, WrappedEnv>(env)
    where OE : Number, OE : Comparable<OE> {
  
  val frames = FixedSizeDeque<NDArray<OE>>(k) { it.unref() }
  override val observation_space = run {
    val (height, width, rgb) = env.observation_space.shape
    Box(0, 255, Shape(height, width, rgb * k),
        env.observation_space.dtype)
  }
  
  override fun reset(): NDArray<OE> {
    native {
      val ob = env.reset()
      repeat(k) {
        frames += ob.copy().ref()
      }
      return get_ob().ref()
    }
  }
  
  override fun step(a: Int): t4<NDArray<OE>, Float, Boolean, Map<String, Any>> {
    native {
      val result = env.step(a)
      frames += result._1.ref()
      result._1 = get_ob().ref()
      return result
    }
  }
  
  private fun get_ob(): NDArray<OE> {
    require(frames.size == k)
    val frames = frames.toList()
    return concat(frames, axis = 2)
  }
}

//class LazyFrames<T : Any>(val frames: List<NDArray<T>>) : NDArray<T>() {
//  lateinit var out: NDArray<T>
//  private fun force(): NDArray<T> {
//    if (!::out.isInitialized) {
//      out = concatenate(frames, axis = 2)
//    }
//    return out
//  }
//
//}