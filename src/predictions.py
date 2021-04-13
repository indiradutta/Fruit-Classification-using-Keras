from model import model
import numpy as np
test_img = 'Avocado.jpg'
img = image.load_img(test_img, target_size = (300,300))
img = image.img_to_array(img, dtype=np.uint8)
img = np.array(img)/255.0
prediction = model.predict(img[np.newaxis, ...])
probability = np.max(prediction[0], axis=-1)
predicted_class = class_names[np.argmax(prediction[0], axis=-1)]