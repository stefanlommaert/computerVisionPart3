# from keras_segmentation.models.unet import vgg_unet

# model = vgg_unet(n_classes = 21, input_height=416,input_width=416)

# model.train(train_images = "prepped_data/train_images/",
#             train_annotations = "prepped_data/train_segmentation/",
#             checkpoints_path = "checkpoints/vgg_unet_1",
#             epochs = 20
#             )


from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes = 21, input_height=416,input_width=416)

model.train(train_images = "prepped_data/train_images/",
            train_annotations = "prepped_data/train_segmentation/",
            checkpoints_path = "checkpoints/vgg_unet_1",
            epochs = 20
            )