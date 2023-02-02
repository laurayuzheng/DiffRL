import os
import shutil
from PIL import Image
from zipfile import ZipFile
import cv2

path = 'icml2023_outputs/pacecar/single_pace_car_video_2023_01_31'

def make_video(path: str):

    file_list = os.listdir(path)
    if any([('image' in file_name) for file_name in file_list]):

        zip_file_name = os.path.join(path, 'images.zip')
        with ZipFile(zip_file_name, "r") as zo:
            tmp_path = os.path.join(path, "tmp")
            zo.extractall(tmp_path)

        file_list = os.listdir(tmp_path) 
        image_name_dict = {}
        for file_name in file_list:
            if 'image' in file_name:
                frame = int(file_name.split('_')[1].split('.')[0])
                image_name_dict[frame] = os.path.join(tmp_path, file_name)

        frame_list = list(image_name_dict.keys())
        frame_list.sort()

        first_image = cv2.imread(image_name_dict[frame_list[0]])
        image_height, image_width, image_layers = first_image.shape

        scale = 0.5
        image_height = int(image_height * scale)
        image_width = int(image_width * scale)

        video_path = os.path.join(path, 'video.avi')
        video = cv2.VideoWriter(video_path, 0, 30., (image_width, image_height))
        for frame in frame_list:
            image_name = image_name_dict[frame]
            image = cv2.imread(image_name)
            image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)
            video.write(image)
        cv2.destroyAllWindows()
        video.release()
        shutil.rmtree(tmp_path)
        
        print("Done {}".format(path))

    for file_name in file_list:
        full_name = os.path.join(path, file_name)
        if os.path.isdir(full_name):
            make_video(full_name)

make_video(path)