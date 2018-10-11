package wumo.sim.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(
        value = @Platform(
                include = {"ale/common/Constants.h",
                        "ale/ale_c_wrapper.h"},
                link = {"ale_c"}
        ),
        target = "wumo.sim.ale",
        helper = "wumo.sim.helper.ale"
)
public class ale implements InfoMapper {

    @Override
    public void map(InfoMap infoMap) {
//        infoMap.put(new Info("ALEInterface").base("Pointer"))
//                .put(new Info("ALEState").pointerTypes("Pointer"));
    }
}
