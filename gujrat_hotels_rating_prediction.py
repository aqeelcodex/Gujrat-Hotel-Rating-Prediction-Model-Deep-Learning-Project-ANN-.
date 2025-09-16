import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Flatten, Concatenate, Input
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# importing the data set
hotel = pd.read_csv("final_hotel.csv")
hotel.drop(columns=["hotel name", "Unnamed: 9", "Unnamed: 10"], inplace=True)
hotel.columns = hotel.columns.str.replace(" ", "_")
print(hotel.head())
print(hotel.isnull().sum())
cat_col = hotel.select_dtypes(include="object").columns
for col in cat_col:
    print(f"{col}: {hotel[col].nunique()}")

num_col = hotel.select_dtypes(exclude="object").columns
for col in num_col:
    scale = StandardScaler()
    hotel[col] = scale.fit_transform(hotel[[col]])

target = "rating_text"
le_target = LabelEncoder()
hotel[target] = le_target.fit_transform(hotel[target])

embd_columns = ["place", "near_by_place", "facilities", "destination_name"]
encoders = {}
for col in embd_columns:
    if col not in hotel.columns:
        raise ValueError(f"Column is not found: {col}")
    
    label = LabelEncoder()
    hotel[col] = label.fit_transform(hotel[col])
    encoders[col] = label
print(hotel.dtypes)

embd_columns = ["place", "near_by_place", "facilities", "destination_name"]
vocabory_sizes = {col: hotel[col].nunique() for col in embd_columns}
print(hotel[col],":", vocabory_sizes)

emb_input = []
emb_layer = []
for col in embd_columns:
    vocab_size = hotel[col].nunique() + 1
    embd_dim = min(50, vocab_size // 2) # Best rule to find the embd dim 

    inpt = Input(shape=(1,), name=f"{col}_input")
    emd = Embedding(input_dim=vocab_size, output_dim=embd_dim, name= f"{col}_emb")(inpt)
    emd = Flatten()(emd)

    emb_input.append(inpt)
    emb_layer.append(emd)

x = Concatenate()(emb_layer) 
num_col = ["rating", "discount_price", "actual_price"]
inpts = Input(shape= (len(num_col),), name= "numeric_inputs")
x = Concatenate()([x, inpts])
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
outputs = Dense(len(le_target.classes_), activation="softmax")(x)

model = Model(inputs= emb_input + [inpts], outputs= outputs)
model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

print("✅ Model built successfully!")

X = hotel.drop(columns=["rating_text"])
y = hotel["rating_text"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_inputs = [X_train[col].values for col in embd_columns] + [X_train[num_col].values]
X_test_inputs = [X_test[col].values for col in embd_columns] + [X_test[num_col].values]

model.fit(X_train_inputs, y_train, epochs= 10, batch_size=32, validation_split=0.2)
y_pred_train = model.predict(X_train_inputs)
y_pred_test = model.predict(X_test_inputs)

y_pred_train = np.argmax(y_pred_train, axis=1)
y_pred_test = np.argmax(y_pred_test, axis=1)

print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))