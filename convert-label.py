import xml.etree.ElementTree as ET
import os
import json

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

image_set = set()

image_id = 20200000000
annotation_id = 0

def addCatItem(name, value):
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item['id'] = value
    category_item['name'] = name
    coco['categories'].append(category_item)
    return

def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    #left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    #right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    #right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def clip(val):
    return max(min(1.0, val), 0.0)

def parseXmlFiles(xml_path, text_dir): 
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue
        
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None
        xml_file = os.path.join(xml_path, f)
        # print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))
        text_name = f.replace('.xml', '.txt')
        text_lines = ""
        #elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            
            current_parent = elem.tag
            current_sub = None
            object_name = None
            
            if elem.tag == 'folder':
                continue
            
            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')
                
            #add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    # print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name)) 
            #subelem is <width>, <height>, <depth>, <name>, <bndbox>
            skip_flg = False
            for subelem in elem:
                bndbox ['xmin'] = None
                bndbox ['xmax'] = None
                bndbox ['ymin'] = None
                bndbox ['ymax'] = None
                
                current_sub = subelem.tag
                if SKIP_DIFFICULT and current_parent == 'object' \
                            and subelem.tag == 'difficult' and subelem.text == '1':
                    skip_flg = True
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        print('skip object: ', object_name, ' in ', xml_file)
                        skip_flg = True
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                #option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                #only after parse the <object> tag
                if bndbox['xmin'] is not None and not skip_flg:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    #x
                    bbox.append(bndbox['xmin'])
                    #y
                    bbox.append(bndbox['ymin'])
                    #w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    #h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    # print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)
                    # text_lines += '{} {} {} {} {} {} {}\n'.format(object_name.replace(' ', '-'),
                    #                             0, 0,
                    #                             bndbox['xmin'], bndbox['ymin'],
                    #                             bndbox['xmax'], bndbox['ymax'])
                    # TODO
                    c_x , c_y = (bndbox['xmin'] + bndbox['xmax']) / 2. / 1280., (bndbox['ymin'] + bndbox['ymax']) / 2. / 960.
                    c_w , c_h = (bndbox['xmax'] - bndbox['xmin']) / 1280., (bndbox['ymax'] - bndbox['ymin']) / 960.
                    text_lines += '{} {} {} {} {}\n'.format(category_set[object_name],
                                                clip(c_x) , clip(c_y), clip(c_w) , clip(c_h))
        with open(os.path.join(text_dir, text_name), 'w') as wf:
            wf.write(text_lines)

if __name__ == '__main__':
    # dataname = 'ir'
    dataname = 'baiguang'
    # split = 'train'
    split = 'val'

    assert dataname in ['baiguang', 'ir']
    assert split in ['train', 'val']

    if dataname == 'baiguang':
        category_set = {
            "wire":0
            , "pet feces":1
            , "shoe":2
            , "bar stool a":3
            , "fan":4
            , "power strip":5
            , "dock(ruby)":6
            , "dock(rubys+tanosv)":7
            , "bar stool b":8
            , "scale":9
            , "clothing item":10
            , "cleaning robot":11
            , "fan b":12
            , "door mark a":13
            , "door mark b":14
            , "wheel":15
            , "door mark c":16
            , "flat base":17
            , "whole fan":18
            , "whole fan b":19
            , "whole bar stool a":20
            , "whole bar stool b":21
            , "fake poop a":22
            , "fake poop b":23
            , "dust pan":24
            , "folding chair":25
            , "laundry basket":26
            , "handheld cleaner":27
            , "sock":28
        }
    # elif dataname == 'ir':
    #     category_set = {
    #         'wire':1
    #         ,'pet feces':2
    #         ,'shoe':3
    #         ,'bar stool a':4
    #         ,'fan':5
    #         ,'power strip':6
    #         ,'dock(ruby)_ir':7
    #         ,'dock(rubys+tanosv)_ir':8
    #         ,'bar stool b':9
    #         ,'scale':10
    #         ,'clothing item':11
    #         ,'cleaning robot':12
    #         ,'fan b':13
    #         ,'door mark a':14
    #         ,'door mark b':15
    #         ,'wheel':16
    #         ,'door mark c':17
    #         ,'flat base':18
    #         ,'whole fan':19
    #         ,'whole fan b':20
    #         ,'whole bar stool a':21
    #         ,'whole bar stool b':22
    #         ,'fake poop a':23
    #         ,'dust pan':24
    #         ,'folding chair':25
    #         ,'laundry basket':26
    #         ,'handheld cleaner':27
    #         ,'sock':28
    #         ,'fake poop b':29
    #     }

    if dataname == 'baiguang' and split == 'train':
        xml_path = '/workspace/downloads/rockrobo_data/det_trainset/VOC2007/Annotations'
        # json_file = '/workspace/downloads/rockrobo_data/det_trainset/VOC2007/instances.json'
        txt_dir = '/workspace/yolov5/baiguang/labels/train/'
        SKIP_DIFFICULT = False
    elif dataname == 'baiguang' and split == 'val':
        xml_path = '/workspace/downloads/rockrobo_data/det_testset/neice_final/VOC2007/Annotations'
        # json_file = '/workspace/downloads/rockrobo_data/det_testset/neice_final/VOC2007/instances.json'
        txt_dir = '/workspace/yolov5/baiguang/labels/val/'
        SKIP_DIFFICULT = True
    # elif dataname == 'ir' and split == 'val':
    #     xml_path = '/workspace/downloads/rockrobo_data/det_testset/shtest_ir/VOC2007/Annotations'
    #     json_file = '/workspace/downloads/rockrobo_data/det_testset/shtest_ir/VOC2007/instances.json'
    #     txt_dir = '/workspace/downloads/rockrobo_data/det_testset/shtest_ir/VOC2007/ann-text'
    #     SKIP_DIFFICULT = True
    # elif dataname == 'ir' and split == 'train':
    #     xml_path = '/workspace/downloads/rockrobo_data/rio_ir/VOC2007/Annotations'
    #     json_file = '/workspace/downloads/rockrobo_data/rio_ir/VOC2007/instances.json'
    #     txt_dir = '/workspace/downloads/rockrobo_data/rio_ir/VOC2007/ann-text'
    #     SKIP_DIFFICULT = False
    else:
        raise Exception('error path choice')

    # convert
    for k, v in category_set.items():
        addCatItem(k, v)
    parseXmlFiles(xml_path, txt_dir)
    # json.dump(coco, open(json_file, 'w'))
    print('convert from: ', xml_path , ' to: ', txt_dir)
    print('complete!!!')

