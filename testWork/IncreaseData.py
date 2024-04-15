import albumentations as A
import numpy as np
from PIL import Image, ImageOps
import cv2

spatial_transform = A.Compose([
            # A.RandomResizedCrop(patch_size * 20, patch_size * 20, scale=(0.5, 1)),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=.1),
                    # A.PiecewiseAffine(p=0.3),
                    ], p=0.2)
            ])

pixel_transform = A.Compose([
            A.GaussNoise(p=0.2),
            A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
            A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                    ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
Nor = A.Compose([
    A.Normalize()
])

image = cv2.imread("./data/27.jpg")
# image = ImageOps.exif_transpose(image)
# image = np.array(image)
# print(f"==>> image.shape: {image.shape}")

blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
# cv2.imwrite("image.jpg", image)
cv2.imwrite("GaussianBlur.jpg", image)
s_img = spatial_transform(image = blurred_image)['image']
cv2.imwrite("spatial_transform.jpg", s_img)
p_img = pixel_transform(image = s_img)['image']
cv2.imwrite("pixel_transform.jpg", p_img)
mean, std = cv2.meanStdDev(p_img)
p_img[:,:,0] = (p_img[:,:,0] - mean[0][0]) / std[0][0]
p_img[:,:,1] = (p_img[:,:,1] - mean[1][0]) / std[1][0]
p_img[:,:,2] = (p_img[:,:,2] - mean[2][0]) / std[2][0]
cv2.imwrite("nor.jpg", p_img)
