from keras_segmentation.models.unet import resnet50_unet

model = resnet50_unet(n_classes=21, input_height=256, input_width=256)

model.load_weights("checkpoints/resnet_unet_1.39")

model.train(train_images="prepped_data/train_images/",
            train_annotations="prepped_data/train_segmentation/",
            val_images="prepped_data/val_images/",
            val_annotations="prepped_data/val_segmentation/",
            checkpoints_path="checkpoints3/resnet_unet_1",
            epochs=60,
            validate=True)
            

from keras_segmentation.models.unet import vgg_unet

model = vgg_unet(n_classes = 21, input_height=256,input_width=256)

model.train(train_images="prepped_data/train_images/",
            train_annotations="prepped_data/train_segmentation/",
            val_images="prepped_data/val_images/",
            val_annotations="prepped_data/val_segmentation/",
            checkpoints_path="checkpoints2/vgg_unet_1",
            epochs=60,
            validate=True)
