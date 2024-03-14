import numpy as np
import cv2  # 导入OpenCV库

# 假设你已经从 .npz 文件中加载了数据
loaded_data = np.load('/tmp/openai-2024-02-06-00-33-22-770771/samples_16x64x64x3.npz')

# 获取存档文件中的所有键
keys = loaded_data.files
# 遍历键，获取图像数据
for key in keys:
    print(key)

image_data = loaded_data['arr_0']
label_data = loaded_data['arr_1']
output_folder = '../results/'
# 循环处理每张图像
for i, image in enumerate(image_data):
    # 可以选择将图像数据的数据类型从uint8转换为float32，具体取决于需要
    # image = image.astype(np.float32)
    image_filename = f'{output_folder}image_{i}.jpg'
    # 使用OpenCV显示图像（如果你使用PIL，可以使用Image.show()方法）
    # print(image)
    # print(type(image))
    # print(image.shape)
    cv2.imwrite(image_filename, image)

# 等待用户按下任意键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
