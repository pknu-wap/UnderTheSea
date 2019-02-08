import os
import csv

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils

class FishDataset(utils.Dataset):
	def __init__(self, config):
    	self.dataset_name = config.NAME

    def clean_name(file_name):
    	file_name = file_name[:-6]
		name = file_name.replace("_", " ")
		name = name.replace("~", " ")
		name = name.rstrip()

		return name

	def clean_dataset(file_list):
		file_names = list()
		for file_name in file_list:
			name = clean_name(file_name)
			file_names.append(name)

		return file_names

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

	def count_each_class(file_names, class_names):
		count = list()
		for i in class_names:
			count.append(file_names.count(i))

		return count

	#image setting 
	def load_mask(image_id, detect_result):
    	"""
    	image_info = self.image_info[image_id]
    	if image_info["source"] not in self.class_names:
        	print("warning: source {} not part of our classes, delegating to parent.".format(image_info["source"]))
        	return super(self.__class__, self).load_mask(image_id)
    	"""
    	mask = detect_result['masks']
    	class_ids = detect_result['class_ids']

    	return mask, class_ids



def create_datasets(dataset_dir, config, train_pct=.8, randomize=True):
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
	# 1. dataset_dir 에서 파일 이름읽어오기
	file_list = os.listdir(dataset_dir)
	file_list.sort()
	del file_list[0]

	# 파일 이름에서 .jpg, _ 지우기
	file_names = FishDataset.clean_dataset(file_list)
	
	# 파일 이름에서 클래스 이름 가져오기 
	class_names = FishDataset.get_unique_classnames(file_names)

	# 2. class별 개수 세기
	count = count_each_class(file_names, class_names)

	# csv 파일 만들어 놓기 
	save_path = os.path.join(ROOT_DIR, "/species.csv")
	f = open(save_path, 'w', encoding='utf-8', newline='')
	writer = csv.writer(f)
	writer.writerows(zip(class_names, count))
	f.close()

	# 3. class별 할당 train 개수 세기
	train_n = list()

	for i in range(len(count)):
  		if(count[i] <5):
   			n = math.floor(count[i]*train_pct)
    		train_n.append(n)
  		else:
    		n = math.ceil(count[i]*train_pct)
    		train_n.append(n)
  		print(train_n)

  	# 4. class 이동하면서 random 숫자 생성
  	train_index = list()

  	for i in range(len(count)):
  		choice = random_choice(count[i], train_n[i])
  		train_index.append(choice)

  	val_index = generate_val_index(count, train_index)

  	# 5. class 이름에서 파일 이름 생성
  	train_set = list()
  	val_set = list()
  	for i in range(len(class_names)):
  		file_set = generate_filename(class_names[i], train_index[i])
  		train_set.append(file_set)
  		file_set = generate_filename(class_names[i], val_index[i])
  		val_set.append(file_set)

  	# 6. train/ val 에 저장하기
  	return train_set, val_set


import random
def random_choice(count, train_n):
	"""
		INPUT
			count = class별 개수
			train_n = train나눌 개수 
		OUTPUT
			choice = 선택된 index

		train_n 수만큼 count 개수 내 랜덤한 숫자 만들기 
	"""
	choice = list()
  
	for i in range(train_n):
		number = random.randrange(1, count+1)
    	while number in choice:
      		number = random.randrange(1, count+1)
    	choice.append(number)
    choice.sort()
    
    return choice
  
def generate_val_index(count, train_index):
	val_index = list()

	for i in range(len(count)):
	no_choice = list()
	for j in range(1, count[i]+1):
		if j not in train_index[i]:
      	no_choice.append(j)
  	val_index.append(no_choice)

  	return val_index

def generate_filename(class_names, index):
  full_names = list()
  
  for i in range(len(index)):
    name = class_names.replace(" ", "_")
    name = "%s_%d.jpg"%(name, index[i])
    full_names.append(name)
    
  return full_names
    