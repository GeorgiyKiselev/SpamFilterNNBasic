# %%
from tensorflow import keras
from keras.preprocessing.text import Tokenizer 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
model = keras.models.load_model('spam_det.h5')

# %%
df = pd.read_csv('spam_ham_dataset.csv')
df = df[['text','label']]

X = df['text']
y = df['label']
y = y.str.replace('ham','0')
y = y.str.replace('spam','1')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size = 0.2, random_state=1)

tokenizer = Tokenizer(num_words = 1000)

tokenizer.fit_on_texts(train_data)

# %%
sus = input("Please paste the text of the email in question here:")
sus = sus.replace("[^a-z0-9 ]+", "").lower()
sus = sus.replace(',','')
sus = sus.replace("'",'')
sus = sus.replace('.','')
sus_token = tokenizer.texts_to_matrix(sus,mode = 'binary')
sus_result = np.round(model.predict(sus_token), decimals = 0).mean()

if sus_result >=0.9:
    print('The algorithm has detected suspicious elements in the email and marked it as spam. The chance of this email being harmful is:', "{:.2f}%".format(sus_result *100))
else:
    print('The algorithm has not detected any suspicious elements in the email. It has not been marked as spam.The chance of this email being harmful is:',"{:.2f}%".format(sus_result *100))


