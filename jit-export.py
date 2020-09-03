from models.yolo import Model
import cv2 
import numpy as np
import torch
import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''

# model = Model("./models/yolov5s.inconv.yaml", nc=29).to("cpu")
# ckpt = torch.load("./weights/best_yolov5s_robo_inconv.pt", map_location="cpu")  # load checkpoint

# ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
#                     if model.state_dict()[k].shape == v.shape}  # to FP32, filter
# model.load_state_dict(ckpt['model'], strict=False)

# model = Model("./models/yolov5s.inconv.yaml", nc=29).to(device)
# ckpt = torch.load("./weights/best_yolov5s_robo_inconv.pt", map_location=device)  # load checkpoint
# ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
#                     if model.state_dict()[k].shape == v.shape}  # to FP32, filter
# model.load_state_dict(ckpt['model'], strict=True)
# model.eval()

device = torch.device("cpu")
model = Model("./models/yolov5s.inconv.yaml", nc=29).to(device)
ckpt = torch.load("./weights/best_yolov5s_robo_inconv.pt", map_location=device)  # load checkpoint
ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                    if model.state_dict()[k].shape == v.shape}  # to FP32, filter
model.load_state_dict(ckpt['model'], strict=True)
model.eval()


# model = torch.load("./weights/best_yolov5s_robo_inconv.pt", map_location=device)['model'].float()
# model.eval()

model(torch.Tensor(1,3,512,672))
# img = cv2.imread('/workspace/centernet/data/baiguang/images/val/StereoVision_L_990964_10_0_0_5026_D_FakePoop_719_-149.jpeg')
# img = cv2.resize(img, (640, 640))
# img = img - np.expand_dims(np.expand_dims(np.array([123.675, 116.28, 103.53]), 0), 0)
# img = img / 58.395

# print(np.max(img), np.min(img))
# img = np.transpose(img, (2,0,1))
# img = np.expand_dims(img, 0)
# img = img / 255.
# img = torch.from_numpy(img)
# output = model(img.float())
# print(type(output), len(output), type(output[0]), output[0].size(),  type(output[1]), output[1][0].shape)
# print(output[0].shape[1])
# np.save('assets/output0.npy', output[0].detach().numpy())
# np.save('assets/output1.npy', output[1].detach().numpy())
# np.save('assets/output2.npy', output[2].detach().numpy())

########## for multiple input, jiangrong ############
# trace_model = torch.jit.trace(model, (torch.Tensor(1,3,640,640)
#                                     , torch.Tensor(1,3,640,640)
#                                     , torch.Tensor(1,3,640,640)
#                                     , torch.Tensor(1,3,640,640)
#                                     )
#                                 )
# trace_model.save('./weights/yolov5s_jit_multiple_input.pt')
trace_model = torch.jit.trace(model, (torch.Tensor(1,3,512,672), ))
trace_model.save('./weights/best_yolov5s_robo_inconv_jit_relu.pt')
