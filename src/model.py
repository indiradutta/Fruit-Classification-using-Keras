from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def model():

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

  model.fit_generator(train_datagen,epochs=4,steps_per_epoch=2753//32)
  
  return model
