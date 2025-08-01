
import time
import cv2
import paho.mqtt.client as mqtt
import json
import demoji

import sys
import os

import pandas as pd
import numpy as np
from sklearn. feature_extraction. text import CountVectorizer
from sklearn. model_selection import train_test_split
from sklearn. tree import DecisionTreeClassifier
import nltk
import re
import string
import pickle


nltk. download('stopwords')
from nltk. corpus import stopwords
stopword=set(stopwords.words('english'))
stemmer = nltk. SnowballStemmer("english")
data = pd. read_csv("data.csv")
#To preview the data
print(data. head())
data["labels"] = data["class"]. map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data = data[["tweet", "labels"]]
print(data. head())
def clean (text):
    text = str (text). lower()
    text = re. sub('[.?]', '', text)
    text = re. sub('https?://\S+|www.\S+', '', text)
    text = re. sub('<.?>+', '', text)
    text = re. sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re. sub('\n', '', text)
    text = re. sub('\w\d\w', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ". join(text)
    text = [stemmer. stem(word) for word in text. split(' ')]
    text=" ". join(text)
    return text
data["tweet"] = data["tweet"]. apply(clean)
x = np. array(data["tweet"])
y = np. array(data["labels"])
cv = CountVectorizer()
X = cv. fit_transform(x)
#Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Model building
model = DecisionTreeClassifier()
#Training the model
model.fit(X_train,y_train)


with open("GaussianNB.pickle", "wb") as outfile:
#"wb" argument opens the file in binary mode
    pickle.dump(model, outfile)
#Testing the model
y_pred = model.predict (X_test)
y_pred#Accuracy Score of our model
from sklearn. metrics import accuracy_score
print (accuracy_score (y_test,y_pred))
#Predicting the outcome
inp = "fuck"
inp = cv.transform([inp]).toarray()
#print(model.predict(inp))
result = model.predict(inp)
print(result[0])


if(result[0][0]=='O' and result[0][1]=='f' ):
    print("Offencive")
else:
    print("Not Offencive")

      
def on_connect(client, userdata, flags, rc):
    print("Conn ected with result code "+str(rc))
    client.subscribe("india/pune/chat/app/server",)


def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))
    msg_object = json.loads(msg.payload)
    print(msg_object[0]['sender'])
    
    
    emoji = demoji.findall(msg_object[0]['msg'])
    print(type(emoji))
    wrong_imoji = 0
    if "❤️" in emoji:
        wrong_imoji = 1
        print("Heart Found")
    elif "🤬" in emoji:
        wrong_imoji = 1
        print("Abusive Imoji Found")
    if(wrong_imoji == 1):
        client.publish("india/pune/chat/app/"+msg_object[0]['reciver'], "<p style=\"font-size:20px;color:red;\">Wrong Imoji Send by User</p>")
        return
    
    inp = cv.transform([msg_object[0]['msg']]).toarray()
    result = model.predict(inp)
    print(result[0])


    if(result[0][0]=='O' and result[0][1]=='f' ):
        print("Offencive")
        client.publish("india/pune/chat/app/"+msg_object[0]['reciver'], "<p style=\"font-size:20px;color:red;\">Offencive Msg is Blocked</p>")
        return
    client.publish("india/pune/chat/app/"+msg_object[0]['reciver'], msg_object[0]['msg'])
    


broker_address="broker.hivemq.com" 
port=1883
#broker_address="iot.eclipse.org" #use external broker
client = mqtt.Client("ksdfjhskdhsiskdhgjsdguksj") #create new instance

client.on_connect = on_connect
client.on_message = on_message
client.connect(broker_address,port) #connect to broker
#client.subscribe("india/pune/chat/app")
client.on_message = on_message

while True:
    client.loop_start()
    
   