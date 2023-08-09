import os
import time
import numpy as np
import json
import glob


def generate_data(subset="train", debug=False):
	ROOT_DIR = os.getcwd()
	json_path_dict = {
		"train": "train/",
		"val": "val/",
		"test": "test/",
	}

	subset_dir = os.path.join(ROOT_DIR, "gtFine", json_path_dict[subset])
	json_files_dir = glob.glob(subset_dir + "*/*.json")

	#if instance is needed
	# instance_count = 0
	c = 0
	json_final_data = []
	for json_file in json_files_dir:
		c += 1
		with open(json_file, 'r') as f:
			data = json.load(f)

		image_id = os.path.basename(json_file[0:-21])
		image_dir = os.path.basename(json_file[0:-35])
		data['image_id'] = image_id
		image_path = os.path.join(os.path.abspath('../'), 'images', json_path_dict[subset], image_dir, image_id + '_leftImg8bit.png')
		data['path'] = image_path
		for i, instance in enumerate(data['objects']):
			## if need change the segmentation,
			# segm = np.array(instance['polygon'])
			# segm = segm.reshape(1, segm.size).tolist()
			# data['objects'][i]['segmentaion'] = segm
			data['objects'][i]['segmentaion'] = data['objects'][i]['polygon']
			data['objects'][i].pop('polygon')
		data['source'] = 'cityscape'
		if debug:
			if image_dir == 'dusseldorf':
				#print(data['objects'])
				print(image_path)
				#print(data['id'])

		json_final_data.append(data)

	with open(subset + '_all.json', 'w') as f:
		json.dump(json_final_data, f)

	print('done')

if __name__ == '__main__':
	generate_data("train", True)
	generate_data("val", True)