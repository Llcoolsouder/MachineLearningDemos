from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# Create Model
model = Sequential()
model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(20, 2000, 1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(8, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

