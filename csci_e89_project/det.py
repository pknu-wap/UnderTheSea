import os
import sys
import json
import numpy as np
import skimage.draw
import collections
import re
import random
import csv
import math
from collections import defaultdict, Counter

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


############################################################
#  Configurations
############################################################

class DetConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """

    def __init__(self, dataset_name, classnames):
        # Give the configuration a recognizable name
        self.dataset_name = dataset_name
        self.NAME = dataset_name

        self.CLASS_NAMES = classnames
        self.ALL_CLASS_NAMES = ['BG'] + self.CLASS_NAMES

        # Number of classes (including background)
        self.NUM_CLASSES = len(self.ALL_CLASS_NAMES)
        self.map_name_to_id = {}
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        Config.__init__(self)

    # A GPU with 12GB memory can fit two images.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    TRAINING_VERBOSE = 1

    TRAIN_BN = False
    #  'relu' or 'leakyrelu'
    ACTIVATION = 'leakyrelu'


############################################################
#  Multi-class Dataset
############################################################
# Example usage:
# Load images from a training directory.
#    dataset_train = DetDataset()
#    dataset_train.load_dataset_images(dataset_path, "train", config.CLASS_NAMES)
#
# Alternatively use the convenience function for taking from one directory and spliting into train and test
#   dataset_train, dataset_val = create_datasets(dataset_path+'/train', config.CLASS_NAMES)
class DetDataset(utils.Dataset):

    def __init__(self, config):
        self.dataset_name = config.NAME
        self.map_name_to_id = {}
        self.actual_class_names = collections.Counter()
        self.class_info = [{"source": "", "id": 0, "name": "BG"},
                               {"source": "", "id": 1, "name": "fish"}]
        utils.Dataset.__init__(self)

    def load_by_annotations(self, dataset_dir, annotations_list, class_names):
        """Load a specific set of annotations and from them images.
        dataset_dir: Root directory of the dataset.
        annotations_list: The annotations (and images) to be loaded.
        class_names: List of classes to use.
        """

        # Find the unique classes and track their count
        for a in annotations_list:
            regions = a['regions']
            for r, v in regions.items():
                object_name = v['region_attributes']['object_name']
                self.actual_class_names[object_name] += 1

        # Add classes. Use class_names to ensure consistency.
        for i, name in enumerate(class_names):
            # Skip over background if it appears in the class name list
            print("class name:", name)
            index = i + 1
            if name != 'BG':
                print('Adding class {:3}:{}'.format(index, name))
                self.add_class(self.dataset_name, index, name)   ## dataset_name, index, class_name
                self.map_name_to_id[name] = index

        # Add images
        for a in annotations_list:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            r_object_name = [r['region_attributes']['object_name'] for r in a['regions'].values()]

            assert len(polygons) == len(r_object_name)

            # load_mask() needs the image shape.
            image_path = os.path.join(dataset_dir, a['filename'])

            # The annotation file needs to be pre-processed to save the shape of the image.
            # If it isn't it will have to be read in.
            if 'height' in a and 'width' in a:
                height = a['height']
                width = a['width']
            else:
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

            self.add_image(
                self.dataset_name,     ## source는 데이터셋 이름
                image_id=a['filename'],  # use file name as a unique image id , image_id는 사진 파일 이름
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                r_object_name=r_object_name)
                


    def load_dataset_images(self, dataset_dir, subset, class_names):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        class_names: List of classes to use.
        """

        # Train or validation dataset?
        assert subset in ["train", "val"]

        print(dataset_dir)
        print(subset)

        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Find the unique classes and track their count
        for a in annotations:
            for id, region in a['regions'].items():
                object_name = region['region_attributes']['object_name']
                self.actual_class_names[object_name] += 1

        # Add classes.
        for i, name in enumerate(class_names):
            # Skip over background if it occurs in the
            index = i + 1
            if name != 'BG':
                print('Adding class {:3}:{}'.format(index, name))
                self.add_class('fish', index, name)

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                self.dataset_name,
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_names: a 1D array of class IDs of the instance masks.
        """
        # If not an object in our dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] not in self.class_names:
            print("warning: source {} not part of our classes, delegating to parent.".format(image_info["source"]))
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = []
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            try:
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1
                class_id = self.map_name_to_id[info['r_object_name'][i]]
                class_ids.append(class_id)
            except:
                print(image_info)
                raise

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == self.dataset_name:
            return info["path"]
        else:
            print("warning: DetDataSet: using parent image_reference for: ", info["source"])
            super(self.__class__, self).image_reference(image_id)


def split_annotations(dataset_dir, config, train_pct=.8, annotation_filename="annotations.json", randomize=True):
    """ divide up an annotation file for training and validation
    dataset_dir: location of images and annotation file.
    config: config object for training.
    train_pct: the split between train and val default is .8
    annotation_filename: name of annotation file.
    randomize: If true (default) shuffle the list of annotations.
    """

    indexes = {}
    for idx, cn in enumerate(config.CLASS_NAMES):
        indexes[cn] = idx

    # Load annotations
    annotations = json.load(open(os.path.join(dataset_dir, annotation_filename)))
    annotations = list(annotations.values())

    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    if randomize:
        # Randomize the annotations then divide
        np.random.shuffle(annotations)

    # Find the unique classes and track their count
    uniq_class_names = collections.Counter()
    images_classes = {}
    total_classes = 0
    for a in annotations:
        rc = np.zeros(len(config.CLASS_NAMES))
        for id, region in a['regions'].items():
            object_name = region['region_attributes']['object_name']
            uniq_class_names[object_name] += 1
            total_classes += 1
            rc[indexes[object_name]] += 1
        images_classes[a['filename']] = rc

    # Calculate the weights for assigning to buckets,
    # the fewer the greater the weight.
    class_weights = np.zeros(len(config.CLASS_NAMES))
    for cn in uniq_class_names:
        class_weights[indexes[cn]] = total_classes / uniq_class_names[cn]

    # Distribute the annotations into buckets by class
    bucket_of_classes = defaultdict(list)
    for a in annotations:
        # Multiply class count by weights to select which bucket.
        t = images_classes[a['filename']] * class_weights
        selected_class = t.argmax()
        bucket_of_classes[config.CLASS_NAMES[selected_class]].append(a)

    train_ann = []
    val_ann = []
    for k, v in bucket_of_classes.items():
        n_for_train = int(len(v)*train_pct)
        train_ann = train_ann + v[:n_for_train]
        val_ann = val_ann + v[n_for_train:]

    def validate_unique(ann, img_files={}):
        for a in ann:
            filename = a['filename']
            if filename in img_files:
                raise RuntimeError(filename+' already exists')
            else:
                img_files[filename] = 1
        return img_files

    img_files = validate_unique(train_ann)
    img_files = validate_unique(val_ann, img_files)
    assert len(train_ann)+len(val_ann) == len(img_files)

    return train_ann, val_ann


def create_datasets(dataset_dir, config, train_pct=.8):
    """ set up the training and validation training set.
    dataset_dir: location of images and annotation file.
    config: config object that includes list of classes being trained for.
    train_pct: the split between train and val default is .8
    """

    train_ann, val_ann = split_annotations(dataset_dir, config, train_pct=train_pct)

    print(annotation_stats(train_ann))
    print(annotation_stats(val_ann))

    train_ds = DetDataset(config)
    train_ds.load_by_annotations(dataset_dir, train_ann, config.CLASS_NAMES)

    val_ds = DetDataset(config)
    val_ds.load_by_annotations(dataset_dir, val_ann, config.CLASS_NAMES)

    assert len(train_ds.image_info) == len(train_ann) and len(val_ds.image_info) == len(val_ann)

    return train_ds, val_ds


def annotation_stats(annotations):
    # Find the unique classes and track their count
    uniq_class_names = collections.Counter()
    for a in annotations:
        for id, region in a['regions'].items():
            object_name = region['region_attributes']['object_name']
            uniq_class_names[object_name] += 1
    return uniq_class_names


##########################################################################################################
## ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  Mask RCNN을 한 번 돌려서 나온 mask를 가지고 다시 학습시킬 때 필요한 것들 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
##########################################################################################################

class FishDataset(utils.Dataset):
    
    def __init__(self, config, result_masks):
        self.dataset_name = config.NAME
        self.map_name_to_id = {}
        self.actual_class_names = collections.Counter()
        self.result_masks = result_masks
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        utils.Dataset.__init__(self)
    
    ## Mask RCNN에서 나온 결과값을 load_mask 메소드에 파일 명과 함께 전달
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != 'fish':
            print("warning: source {} not part of our classes, delegating to parent.".format(image_info["source"]))
            return super(self.__class__, self).load_mask(image_id)
    
        class_ids = []
        if image_info['id'] in self.result_masks.keys():
            class_name = self.get_class_name(image_info['id'])
            for class_info in self.class_info:
                if class_name == class_info['name']:
                    class_ids.append(class_info['id'])
        
        # Return mask, and array of class IDs of each instance.
        # 마스크를 돌려줄 때, 각 인스턴스의 클래스 아이디 번호를 넘파이 배열로 넘겨줘야한다.
        class_ids = np.array(class_ids, dtype=np.int32)
        return self.result_masks[image_info['id']], class_ids
                
    ## 파일 이름에서 클래스 id 추출
    def get_class_name(self, file_name):
        ## 파일명을 class ID로 사용하기 위해서 정규식으로 잘라냄
        parse = re.sub('[0-9.]', '', file_name)
        parse1 = re.sub('jpg$|jpeg$', '', parse)
        parse2 = re.sub('_', ' ', count=1, string=parse1)
        parse3 = re.sub('_', '', parse2)
        return parse3
    
    ## 데이터셋 형식에 맞춰 만들기
    def make_add_image(self, image_path, file_list):

        for file_list_in_list in file_list:
            for file_name in file_list_in_list:
                img_path = os.path.join(image_path, file_name)
                self.add_image(
                           source = self.dataset_name,
                           image_id = file_name,  # use file name as a unique image id , image_id는 사진 파일 이름
                           path=img_path)
                
    def update_class_names(self, class_names):
         # Add classes. Use class_names to ensure consistency.
        for i, name in enumerate(class_names):
        # Skip over background if it occurs in the
            index = i + 1
            if name != 'BG':
                print('Adding class {:3}:{}'.format(index, name))
                self.add_class(self.dataset_name, index, name)
                self.map_name_to_id[name] = index



def create_datasets_from_result(dataset_dir, config, result_masks, train_pct=.8):
    
    ## get file names from directory
    file_list = get_file_name(dataset_dir)
    ## Make Train list, Validation list
    train_list, val_list = divide_dataset(dataset_dir, file_list)

    print('Train list : ', train_list, '\n')
    print('Validation list : ', val_list, '\n')

    train_ds = FishDataset(config, result_masks)
    train_ds.update_class_names(config.CLASS_NAMES)
    train_ds.make_add_image(dataset_dir, train_list)
    
    
    val_ds = FishDataset(config, result_masks)
    val_ds.update_class_names(config.CLASS_NAMES)
    val_ds.make_add_image(dataset_dir, val_list)
    
    
    train_len = 0
    val_len = 0
    for i in train_list:
        train_len += len(i)

    print('train_len : ',train_len)

    for i in val_list:
        val_len += len(i)
    print('val_len : ',val_len)
    
    assert len(train_ds.image_info) == train_len and len(val_ds.image_info) == val_len
    
    return train_ds, val_ds


def get_unique_classnames(file_names):
    """
        1. 파일 이름 읽어옴
        2. class_names 안에 이름 없으면 추가
        """
    class_names = list()
    for i in file_names:
        if i not in class_names:
            class_names.append(i)
        
        return class_names


def count_each_class(file_list):
    count = Counter()
    for i in file_list:
        parse = re.sub('[0-9.]', '', i)
        parse1 = re.sub('jpg$|jpeg$', '', parse)
        parse2 = re.sub('_', ' ', count=1, string=parse1)
        parse3 = re.sub('_', '', parse2)
        count[parse3] += 1
        
    return count
    

def get_class_name(file_list):
    class_ids = set()
    for key in file_list:
        ## 파일명을 class ID로 사용하기 위해서 정규식으로 잘라냄
        parse = re.sub('[0-9.]', '', key)
        parse1 = re.sub('jpg$|jpeg$', '', parse)
        parse2 = re.sub('_', ' ', count=1, string=parse1)
        parse3 = re.sub('_', '', parse2)
        class_ids.add(parse3)
    
    return list(class_ids)


def random_choice(count, train_n):
    """
        INPUT
        count = class별 개수
        train_n = train나눌 개수
        OUTPUT
        choice = 선택된 index
        
        train_n 수만큼 count 개수 내 랜덤한 숫자 만들기
        """
    
    num_list = np.arange(count)+1
    choice = np.random.choice(num_list, train_n, replace=False)
    choice = list(choice)
    choice.sort()
    
    return choice


def generate_val_index(count, train_index):
    ## count는 클래스의 총 갯수
    total = set(np.arange(count)+1)
    train = set(train_index)
    val_index = total-train
        
    return list(val_index)


def generate_filename(class_names, index):
    full_names = []
    for i in range(len(index)):
        name = class_names.replace(" ", "_")
        name = "%s_%d.jpg"%(name, index[i])
        full_names.append(name)
    
    return full_names


def get_file_name(dataset_dir):
    # 1. dataset_dir 에서 파일 이름읽어오기
    file_list = os.listdir(dataset_dir)
    file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
    
    return file_list_jpg

    
def divide_dataset(dataset_dir, file_list, train_pct = 0.8):
    
    """
        dataset_dir 받아서 train/val 파일 이름 나눠주기
        
        file_list / file_names / class_names / count / train_n / train_index
        
        1. dataset_dir 에서 파일 이름 읽어오기
        2. class별 개수 세기
        3. class별 할당 train 개수 세기
        4. class 이동하면서 random 숫자 생성
        5. class 이름에서 파일 이름 생성
        6. train/ val 에 저장하기
        
        return train, val
        """
    
    # 1. 파일 이름에서 class id 만들기
    class_names = get_class_name(file_list)

    # 2. class별 개수 세기
    count = count_each_class(file_list)
    class_list = list(count.keys())
    class_list.sort()

    # 3. class별 할당 train 개수 세기
    train_n = []

    for i in class_list:
        if(count[i] <5):
            n = math.floor(count[i]*train_pct)
            train_n.append(n)
        else:
            n = math.ceil(count[i]*train_pct)
            train_n.append(n)

    # 4. class 이동하면서 random 숫자 생성
    train_index = []
    val_index = []

    for i, key in enumerate(class_list):
        choice = random_choice(count[key], train_n[i])
        train_index.append(choice)
        val_choice = generate_val_index(count[key], choice)
        val_index.append(val_choice)


    # 5. class 이름에서 파일 이름 생성
    train_set = []
    val_set = []
    class_names.sort()
    for i in range(len(train_index)):
        file_set = generate_filename(class_names[i], train_index[i])
        train_set.append(file_set)
        file_set = generate_filename(class_names[i], val_index[i])
        val_set.append(file_set)

    # 6. train/ val 에 저장하기
    return train_set, val_set

                            




##########################################################################################################
## ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑   Mask RCNN을 한 번 돌려서 나온 mask를 가지고 다시 학습시킬 때 필요한 것들 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
##########################################################################################################



if __name__ == "__main__" :
    config = DetConfig('Fish', ['Fish'])
    dataset_train, dataset_val = create_datasets('./images/fish/train', config)

    # config = DetConfig('sign', ['sign', 'yield_sign', 'stop_sign', 'oneway_sign', 'donotenter_sign', 'wrongway_sign'])
    # dataset_train, dataset_val = create_datasets('./images/signs/train', config)

    dataset_train.prepare()
    dataset_val.prepare()

    print("Training Images: {}\nClasses: {}".format(len(dataset_train.image_ids), dataset_train.class_names))
    print("Validation Images: {}\nClasses: {}".format(len(dataset_val.image_ids), dataset_val.class_names))
