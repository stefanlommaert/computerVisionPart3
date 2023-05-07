from keras_segmentation.predict import predict, predict_multiple

class_names = ['bg', "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), 
          (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), 
          (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
          (192, 0, 0), (0,192,0)]

color_dict = {
    "background": (0, 0, 0),
    "aeroplane": (255, 0, 0),
    "bicycle": (0, 255, 0),
    "bird": (0, 0, 255),
    "boat": (255, 255, 0),
    "bottle": (255, 0, 255),
    "bus": (0, 255, 255),
    "car": (128, 0, 0),
    "cat": (0, 128, 0),
    "chair": (0, 0, 128),
    "cow": (128, 128, 0),
    "diningtable": (128, 0, 128),
    "dog": (0, 128, 128),
    "horse": (64, 0, 0),
    "motorbike": (0, 64, 0),
    "person": (0, 0, 64),
    "pottedplant": (64, 64, 0),
    "sheep": (64, 0, 64),
    "sofa": (0, 64, 64),
    "train": (192, 0, 0),
    "tvmonitor": (0, 192, 0)
}


predict( 
	checkpoints_path="checkpoints3/resnet_unet_1.54", 
	inp="prepped_data/val_images/augmented_0_train_0.png", 
	out_fname="outputfinal.png",
	class_names=class_names,
    colors= colors,
	prediction_height=128,
    prediction_width=128
)


# predict_multiple( 
# 	checkpoints_path="checkpoints3/resnet_unet_1.54", 
# 	inp_dir="prepped_data/test_images/", 
# 	out_dir="predicted_images",
#     colors=colors,
#     class_names=class_names,
#     prediction_height=128,
#     prediction_width=128
# )