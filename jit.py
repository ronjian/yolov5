from models.yolo import Model
from utils.utils import *

model = Model("models/yolov5s-rockrobo.yaml", ch=3, nc=29).to(device = torch_utils.select_device('cpu'))
model.eval()

trace_model = torch.jit.trace(model, torch.Tensor(1,3,640,640))

trace_model.save('./weights/yolov5s_jit.pt')
print('done')