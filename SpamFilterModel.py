# %%
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
#from sklearn.feature_extraction import CountVectorizer

# %%
df = pd.read_csv('spam_ham_dataset.csv')
df = df[['text','label']]

# %%
X = df['text']
y = df['label']
y = y.str.replace('ham','0')
y = y.str.replace('spam','1')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size = 0.2, random_state=1)

# %%
train_data = train_data.str.replace('[^a-zA-Z ]', '')
test_data = test_data.str.replace('[^a-zA-Z ]', '')

# %%
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words = 1000)

tokenizer.fit_on_texts(train_data)
X_train = tokenizer.texts_to_matrix(train_data,mode = 'binary')

#tokenizer.fit_on_texts(test_data)
X_test = tokenizer.texts_to_matrix(test_data,mode = 'binary')

y_train = np.asarray(train_labels).reshape(4136,1)
y_test = np.asarray(test_labels).reshape(1035,1)

# %%
from keras import models, layers

model = models.Sequential()
model.add(layers.Embedding(1000, 64, input_length=1000))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])


# %%
history = model.fit(X_train, y_train, epochs = 5, batch_size= 8, validation_split = 0.2)

# %%
training_loss = history.history['loss']
validation_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1,len(training_loss)+1)

plt.plot(epochs,training_loss,'bo',label='Training Loss')
plt.plot(epochs,validation_loss,'b',label='Validation Loss')
plt.title('Loss vs Epochs')
plt.legend()
plt.show()

# %%
plt.plot(epochs,accuracy,'bo',label='Training Accuracy')
plt.plot(epochs,val_accuracy,'b',label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()
plt.show()

# %%
model.evaluate(X_test, y_test)

# %%
from sklearn.metrics import confusion_matrix
y_pred = np.round(model.predict(X_test),decimals=0)

confusion_matrix(y_test,y_pred)

# %%
model.save('spam_det.h5')


