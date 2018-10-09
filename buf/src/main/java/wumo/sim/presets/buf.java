package wumo.sim.presets;


import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(
    value = @Platform(
        include = {"buf/buf.h"},
        link = {"buf"}
    ),
    target = "wumo.sim.buf"
)
public class buf implements InfoMapper {
  @Override
  public void map(InfoMap infoMap) {
    String[] types = new String[]{"unsigned char", "char", "short", "int", "long long", "float", "double"};
    
    for (String A : types) {
      for (String B : types) {
        infoMap.put(new Info("cast<" + A + "," + B + ">")
            .javaNames("cast" + A.replace(" ", "") + "2" + B.replace(" ", "")));
      }
    }
  }
}