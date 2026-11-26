
from pyservicemaker import Pipeline, Flow

video_file = (
    "/opt/nvidia/deepstream/deepstream-9.2/"
    "samples/streams/sample_1080p_h264.mp4"
)

pipeline = Pipeline("industrial-safety-test")

Flow(pipeline).capture([video_file]).render()()
