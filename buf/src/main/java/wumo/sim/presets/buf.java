package wumo.sim.presets;


import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(
        value = @Platform(
                include = {"buf/library.h"},
                link = {"buf"}
        ),
        target = "wumo.sim.buf"
)
public class buf implements InfoMapper {
    @Override
    public void map(InfoMap infoMap) {
    }
}