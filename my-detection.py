#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils
from PIL import Image, ImageDraw, ImageFont

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
#camera = jetson.utils.videoSource("/dev/video0")
image_path="/home/nvidia/jetson-inference/examples/imagedata/Ajc3ezCTN4FGz2vF4LpQn9-1200-80.jpg"
img = jetson.utils.loadImage(image_path)
# '/dev/video0' for V4L2
detections = net.Detect(img)
# 打开图片用于绘制
image = Image.open(image_path)
draw = ImageDraw.Draw(image)
# 设置字体（需要系统中有这个字体文件，或者使用默认字体）
font = ImageFont.load_default()

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
     # 绘制边界框
    draw.rectangle((detection.Left, detection.Top, detection.Right, detection.Bottom), outline=(255,0,0), width=2)
    
    # 绘制类别和置信度标签
    label = f"{net.GetClassDesc(detection.ClassID)}: {detection.Confidence:.2f}"
    draw.text((detection.Left, detection.Top - 10), label, fill=(255,0,0), font=font)

# 保存带有标注的图片
output_path = "/home/nvidia/jetson-inference/examples/imagedata/Ajc3ezCTN4FGz2vF4LpQn9-1200-80.jpg_annotated.png"
image.save(output_path)

# 输出保存路径
print(f"Annotated image saved to {output_path}")
    


