from models.yolo import Model

import torch

model = Model("./models/yolov5s.inconv.yaml", nc=29).to("cpu")
ckpt = torch.load("./weights/best_yolov5s_robo_inconv.pt", map_location="cpu")  # load checkpoint

ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                    if model.state_dict()[k].shape == v.shape}  # to FP32, filter
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
model(torch.Tensor(1,3,640,640))

########## for multiple input, jiangrong ############
# trace_model = torch.jit.trace(model, (torch.Tensor(1,3,640,640)
#                                     , torch.Tensor(1,3,640,640)
#                                     , torch.Tensor(1,3,640,640)
#                                     , torch.Tensor(1,3,640,640)
#                                     )
#                                 )
# trace_model.save('./weights/yolov5s_jit_multiple_input.pt')
trace_model = torch.jit.trace(model, (torch.Tensor(1,3,640,640), ))
trace_model.save('./weights/best_yolov5s_robo_inconv_jit.pt')
