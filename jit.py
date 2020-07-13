from models.yolo import Model
from utils.utils import *

model = Model("models/yolov5s-rockrobo.yaml", ch=3, nc=29).to(device = torch_utils.select_device('cpu'))
model.eval()

########## for multiple input, jiangrong ############
# trace_model = torch.jit.trace(model, (torch.Tensor(1,3,640,640)
#                                     , torch.Tensor(1,3,640,640)
#                                     , torch.Tensor(1,3,640,640)
#                                     , torch.Tensor(1,3,640,640)
#                                     )
#                                 )
# trace_model.save('./weights/yolov5s_jit_multiple_input.pt')
trace_model = torch.jit.trace(model, (torch.Tensor(1,3,640,640), ))
trace_model.save('./weights/yolov5s_jit.pt')
########### end #################



print('done')