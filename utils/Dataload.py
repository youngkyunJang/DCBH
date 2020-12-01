import os, pickle
import numpy as np

def check_path_valid(path):
	return path if path.endswith('/') else path + '/'

def load_data_split_pickle(dataset): #you can change this method to load your dataset
	def get_files(vec_folder):
		file_names = os.listdir(vec_folder)
		file_names.sort()
		vec_folder = check_path_valid(vec_folder)
		for i in range(len(file_names)):
			file_names[i] = vec_folder + file_names[i]
		return file_names

	def load_data_xy(file_names):
		datas  = []
		labels = []
		for file_name in file_names:
			with open(file_name, 'rb') as f:
				x, y = pickle.load(f, encoding='bytes')
			datas.append(x)
			labels.append(y)
		data_array = np.vstack(datas)
		label_array = np.hstack(labels)
		return data_array, label_array

	test_folder, train_folder = dataset
	test_file_names = get_files(test_folder)
	train_file_names = get_files(train_folder)
	test_set = load_data_xy(test_file_names)
	train_set = load_data_xy(train_file_names)
	train_set_x, train_set_y = train_set[0], train_set[1]
	test_set_x, test_set_y = test_set[0], test_set[1]

	return [(train_set_x, train_set_y), (test_set_x, test_set_y)]

def deprocess_image(x):
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1
	x = np.clip(x, -1, 1)
	return x