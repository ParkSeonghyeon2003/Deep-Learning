from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.applications.densenet import decode_predictions

# load the model
model = DenseNet121(weights='imagenet')

# load an image from file
image = load_img('data/sample_img_1.jpg', target_size=(224, 224))

# convert the image pixels to a numpy array
image = img_to_array(image)

# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# prepare the image for the model
image = preprocess_input(image)

# predict the probability across all output classes
pred = model.predict(image)

# convert the probabilities to a list of tuples (class, description, probability)
label = decode_predictions(pred)

# retrieve the most likely result, e.g. highest probability
label = label[0][0]

# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))

from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

this_layer = model.layers[2]

tmp_model = Model(inputs=model.input, outputs=this_layer.output)
feature_map = tmp_model.predict(image, verbose=0)

print('Layer Name: ', this_layer.name)
print('feature map shape: ', feature_map.shape)

plt.imshow(feature_map[0, :, :, 0], cmap='gray')
plt.show()

plt.imshow(feature_map[0, :, :, 1], cmap='gray')
plt.show()

# average feature maps
avgmap = feature_map.mean(axis=-1)[0]
plt.imshow(avgmap, cmap='gray')
plt.show()