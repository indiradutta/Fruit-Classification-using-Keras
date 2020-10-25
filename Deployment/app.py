import numpy as np
import streamlit as st
import time
from PIL import Image
from keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train = 'train'
# Create a generator
train_datagen = ImageDataGenerator(
  rescale=1./255
)
train_datagen = train_datagen.flow_from_directory(
        train,
        batch_size=32,
        target_size=(300, 300),
        class_mode='sparse')

labels = train_datagen.class_indices
Labels = '\n'.join(sorted(train_datagen.class_indices.keys()))
with open('Labels.txt', 'w') as file:
  file.write(Labels)
class_names = list(labels.keys())

model=Sequential()
#Convolution blocks
model.add(Conv2D(32, kernel_size = (3,3), padding='same',input_shape=(300,300,3),activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 
model.add(Conv2D(64, kernel_size = (3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 
model.add(Conv2D(32, kernel_size = (3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 
#Classification layers
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6,activation='softmax'))
model.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

def training():
    model.fit_generator(train_datagen,
            epochs=1,
            steps_per_epoch=2753//32)


html_temp_2 = '''
    <div style = "padding-bottom: 20px; padding-top: 20px; padding-left: 20px; padding-right: 20px">      
    <center><h2>Fruit Classifier</h2></center>
    </div>
    '''
st.markdown(html_temp_2, unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding', False)
select = st.selectbox("Please select how you want to upload the image",("Please Select","Upload image via link","Upload image from device"))
if select == "Upload image via link":
    try:
        img = st.text_input('Enter the Image Address')
        img = Image.open(urllib.request.urlopen(img))
    except:
        if st.button('Submit'):
            show = st.error("Please Enter a valid Image Address!")
            time.sleep(4)
            show.empty()

if select == 'Upload image from device':
    file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is not None:
        img = Image.open(file)

try:
    if img is not None:
        st.image(img, width = 300, caption = 'Uploaded Image')
        if st.button('Predict'):
            img = np.array(img.resize((300, 300), Image.ANTIALIAS))
            img = np.array(img, dtype='uint8')
            img = np.array(img)/255.0
            training()
            prediction = model.predict(img[np.newaxis, ...])
            predicted_class = class_names[np.argmax(prediction[0], axis=-1)]
            st.success("Classified Class is : {}".format(predicted_class))

except:
    pass