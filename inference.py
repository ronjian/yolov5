from models.yolo import Model
from utils.utils import *
import cv2
import numpy as np
import torchvision

class_cnt = 29
an_vec = class_cnt + 1 + 4
an_c = an_vec * 3
image_width = 640
CSUM=1

classnames = ["__background__", "wire", "pet feces", "shoe", "bar stool a", "fan", "power strip", "dock(ruby)",
                 "dock(rubys+tanosv)", "bar stool b", "scale", "clothing item", "cleaning robot",
                 "fan b", "door mark a", "door mark b", "wheel", "door mark c", "flat base", "whole fan",
                 "whole fan b", "whole bar stool a", "whole bar stool b", "fake poop a", "fake poop b", "dust pan",
                 "folding chair", "laundry basket", "handheld cleaner", "sock"]

colors = [None
, (0  , 255, 0)
,(255, 0  , 0)
,(0  , 0  , 255)
,(255, 0  , 255)
,(0  , 255, 255)
,(255, 255, 0)
,(0  , 127, 0)
,(0  , 0, 127)
,(127, 0, 0)
,(127, 0  , 255)
,(0  , 127, 255)
,(100, 50 , 200)
,(0  , 127, 255)
,(127, 255, 255)
,(255, 127, 255)
,(255, 255, 127)
,(127, 127, 255)
,(127, 255, 127)
,(255, 127, 127)
,(127, 100, 127)
,(100, 127, 127)
,(127, 127, 100)
,(100, 127, 50) 
,(50 , 200, 127)
,(255, 100, 127)
,(100, 255, 127)
,(255, 50 , 127)
,(200, 255, 127)
,(200, 100, 127)
]

def clip(val):
    return min(max(val, 0.0), 1.0)

def parse_det(an, an_h, an_w, an_s, an_wh):
    global CSUM
    np.save("./assets/output_rk/{}.npy".format(CSUM), an.view(1, 3,an_h,an_w,34).permute(0, 1, 3, 4, 2).contiguous().detach().numpy())
    CSUM += 1
    an = an.flatten().tolist()
    res = []
    for h in range(an_h):
        for w in range(an_w):
            for c in range(an_c):
                val = an[int(c * an_h * an_w + h * an_w + w)]
                if c % an_vec == 0:
                    c_x, c_y, b_w, b_h, obj_conf, score = val, None, None, None, None, None
                    c_x = (c_x * 2. - 0.5 + w) * an_s
                elif c % an_vec  == 1:
                    c_y = val
                    c_y = (c_y * 2. - 0.5 + h) * an_s
                elif c % an_vec == 2:
                    b_w = val
                    b_w = (b_w * 2.) ** 2. * an_wh[c // an_vec][0]
                elif c % an_vec == 3:
                    b_h = val
                    b_h = (b_h * 2.) ** 2. * an_wh[c // an_vec][1]
                elif c % an_vec == 4:
                    obj_conf = val
                else:
                    # if obj_conf and obj_conf > 0.3 and val > 0.5:
                    if obj_conf > 0.001:
                        score = obj_conf * val
                        # print(an_h, an_w, h, w, c % an_vec, c // an_vec, c_x, c_y, b_w, b_h, obj_conf, val, score)
                        res.append((c_x, c_y, b_w, b_h, score, int(c % an_vec-4)))
    return res   

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model("./models/yolov5s.inconv.yaml", nc=29).to(device)
    ckpt = torch.load("./weights/best_yolov5s_robo_inconv.pt", map_location=device)  # load checkpoint

    ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                        if model.state_dict()[k].shape == v.shape}  # to FP32, filter
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    val_dir = '/workspace/centernet/data/baiguang/images/val/'
    for imname in  os.listdir(val_dir):
        if imname.endswith('.jpeg') and imname == "StereoVision_L_1466362_46_0_1_1401_D_FakePoop_1492_1890.jpeg":
        # if imname.endswith('.jpeg'):
            print(imname)
            origin_img = cv2.imread(val_dir + imname)
            origin_height, origin_width, _ = origin_img.shape
            image_height = int(image_width / origin_width * origin_height)
            pad_top = (image_width - image_height) // 2
            pad_bottom = image_width - image_height - pad_top
            input_img = cv2.resize(origin_img, (image_width, image_height))
            input_img = cv2.copyMakeBorder(input_img
                                                , top = pad_top
                                                , bottom = pad_bottom
                                                , left=0
                                                , right=0
                                                , borderType=cv2.BORDER_CONSTANT
                                                , value=(114, 114, 114))  # add border
            assert input_img.shape == (image_width, image_width, 3)
            input_img = np.transpose(input_img, (2,0,1))
            input_img = np.expand_dims(input_img, 0)
            input_img = input_img / 255.
            # print(input_img)
            input_img = torch.from_numpy(input_img)
            input_img = input_img.float()
            input_img = input_img.to(device)
            outputs = model(input_img)

            # print(type(outputs), len(outputs), outputs[0].shape, outputs[1].shape, outputs[2].shape)

            R = []
            R += parse_det(outputs[2]
                    , 20
                    , 20
                    , 32
                    , [[214,99], [287,176], [376,365]])

            R += parse_det(outputs[1]
                    , 40
                    , 40
                    , 16
                    , [[94,219], [120,86], [173,337]])

            R += parse_det(outputs[0]
                    , 80
                    , 80
                    , 8
                    , [[28,31], [53,73], [91,39]] )
            boxes = []
            scores = []
            classids = []
            for box in R:
                c_x, c_y, b_w, b_h, score, classid = box
                score = round(score, 4)
                x0 = int(clip((c_x - b_w / 2.) / image_width) * origin_width)
                y0 = int(clip((c_y - b_h / 2. - pad_top) / image_height) * origin_height)
                x1 = int(clip((c_x + b_w / 2.) / image_width) * origin_width)
                y1 = int(clip((c_y + b_h / 2. - pad_top) / image_height) * origin_height)
                # classname = classnames[classid].replace(' ','-')
                boxes.append([x0, y0, x1, y1])
                scores.append(score)
                classids.append(classid)
            boxes = np.array(boxes)
            scores = np.array(scores)
            classids = np.array(classids)
            # print(boxes.dtype, scores.dtype, classids.dtype)
            sel = torchvision.ops.boxes.nms(torch.from_numpy(boxes).float()
                                        , torch.from_numpy(scores).float()
                                        , 0.65)
            sel = list(sel.numpy())
            boxes = boxes[sel]
            scores = scores[sel]
            classids = classids[sel]
            with open('./assets/txt/' + imname.replace('jpeg', 'txt'), 'w') as wf:
                for box, score, classid in zip(boxes, scores, classids):
                    x0, y0, x1, y1 = box
                    classname = classnames[classid].replace(' ','-')
                    color = colors[classid]
                    wf.write("{} {} {} {} {} {}\n".format(classname, score, x0, y0, x1, y1))
                    if score > 0.2:
                        origin_img = cv2.rectangle(origin_img, (x0, y0), (x1, y1), color, thickness=3)

            cv2.imwrite('./assets/vis/'+imname, origin_img)
