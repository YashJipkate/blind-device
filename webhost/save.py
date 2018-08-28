from keras.applications import vgg16

model = vgg16.VGG16(weights = 'imagenet')

model.save('vgg16_model.h5')

