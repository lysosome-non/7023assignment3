#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
#camera = jetson.utils.videoSource("/dev/video0")
image_path="/home/nvidia/jetson-inference/examples/imagedata/image21.png"
img = jetson.utils.loadImage(image_path)
# '/dev/video0' for V4L2
detections = net.Detect(img)
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
#display = net.Detect
#while display.IsStreaming(): # main loop will go here
if img is not None and detections is not None:
    #img = camera.Capture()
    #if img is None: # capture timeout
        #continue
    #detections = net.Detect(img)
    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    for detection in detections:
        print(f"ClassID:{detection.ClassID}")
        print(f"Confidence:{detection.Confidence}")
        print(f"Left:{detection.Left}")
        print(f"Top:{detection.Top}")
        print(f"Right:{detection.Right}")
        print(f"Bottom:{detection.Right}")
        print(f"Width:{detection.Width}")
        print(f"Height:{detection.Height}")
        print(f"Area:{detection.Area}")
        print(f"Center:{detection.Center}")
    


