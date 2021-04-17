import tarfile
import os
import os.path

def ensure_folder(folder):
	if not os.path.isdir(folder):
		os.mkdir(folder)

data_folder = 'tarfiles'
write_folder = 'landsat8_data'
data_list = os.listdir(data_folder)

for i in range(len(data_list)):

	filename = data_list[i]

	tar_ball = tarfile.open('tarfiles/' + filename)

	tar_folder = filename[:-4]

	ensure_folder(write_folder)
	ensure_folder(write_folder + '/' + tar_folder)

	tar_ball.extractall(write_folder + '/' + tar_folder)
	tar_ball.close()