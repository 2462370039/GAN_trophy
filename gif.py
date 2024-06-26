import imageio
import os

import os
import imageio

# 指定图片文件夹路径
image_folder = 'out\\images'
# 定义 GIF 文件的保存路径和名称
gif_path = 'out\\output.gif'

# 处理文件名
files_path = os.listdir(image_folder)
image_files = [os.path.splitext(file)[0] for file in files_path]
image_files = [int(file) for file in image_files]
image_files.sort()
image_files = [str(file) for file in image_files]
image_files = [file + '.png' for file in image_files]
image_files = [os.path.join(image_folder, file) for file in image_files]
print(image_files)

# 设置每帧的延迟时间（以秒为单位）
frame_duration = 0.5

# 读取图片文件并将其添加到帧列表中
frames = []
for file_path in image_files:
    frames.append(imageio.imread(file_path))

# 使用 imageio 创建 GIF 并设置每帧的延迟时间
with imageio.get_writer(gif_path, mode='I', duration=frame_duration) as writer:
    for frame in frames:
        writer.append_data(frame)