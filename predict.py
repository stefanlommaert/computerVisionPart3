from keras_segmentation.predict import predict


predict( 
	checkpoints_path="checkpoints/resnet_unet_1.59", 
	inp="prepped_data/test_images/test_12.png", 
	out_fname="output.png" 
)