import os
from PIL import Image

path = 'icml2023_outputs/pacecar/single_pace_car_video_2023_01_31'

def make_video(path: str):

    file_list = os.listdir(path)
    if any([('image' in file_name) for file_name in file_list]):

        image_name_dict = {}
        for file_name in file_list:
            if 'image' in file_name:
                frame = int(file_name.split('_')[1].split('.')[0])
                image_name_dict[frame] = os.path.join(path, file_name)

        frame_list = list(image_name_dict.keys())
        frame_list.sort()

        #image_name_list.sort()
        image_list = []

        for frame in frame_list:
            image_name = image_name_dict[frame]
            image = Image.open(image_name)
            image_list.append(image)

        image_list[0].save(os.path.join(path, "video.gif"), save_all=True, append_images=image_list[1:], duration=30, loop=True)

        print("Done {}".format(path))

    for file_name in file_list:
        full_name = os.path.join(path, file_name)
        if os.path.isdir(full_name):
            make_video(full_name)

make_video(path)