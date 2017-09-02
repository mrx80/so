import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from time import time
from datetime import datetime
import tensorflow as tf
from tensorflow.python.ops.variables import global_variables_initializer

df = pd.read_csv('titanic_train.csv')

#sanitize age and sex - fill out missing data, convert strings to zeros
femalemed = df[df['Sex'] == 'female']['Age'].median()
malemed = df[df['Sex'] == 'male']['Age'].median()
df['Age'] = df['Age'].replace(df[(df['Sex']=='male') & (df['Age'].isnull())]['Age'], malemed)
df['Age'] = df['Age'].replace(df[(df['Sex']=='female') & (df['Age'].isnull())]['Age'], femalemed)
df['Sex'] = [0 if x=='male' else 1 for x in df['Sex']]

#my own features
df['fam'] = df['SibSp'] + df['Parch']

#encode embarked into a number
catenc = pd.factorize(df['Embarked'])
df['embarked_enc'] = catenc[0]

df_train = df.sample(frac=0.2)
df_test = df.drop(df_train.index)

scaler = StandardScaler()

features = ['Pclass', 'Sex', 'Age', 'Fare', 'fam', 'embarked_enc']

X_train = scaler.fit_transform(df_train[features].values)
y_train = df_train['Survived'].values
y_train_onehot = pd.get_dummies(df_train['Survived']).values

X_test = scaler.transform(df_test[features].values)
y_test = df_test['Survived'].values


def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(random_state=0, verbose=0)
    model = model.fit(X_train, y_train)
    
    y_prediction = model.predict(X_test)
    return np.sum(y_prediction == y_test) / float(len(y_test))

print randomforest()


mytrain = df.sample(frac=0.8)
mytest = df.drop(mytrain.index)

scaler = StandardScaler()

features = ['Pclass', 'Sex', 'Age', 'Fare', 'fam', 'embarked_enc']
#features = ['Pclass', 'Sex', 'Age', 'Fare']

X_train = scaler.fit_transform(mytrain[features].values)
y_train = mytrain['Survived'].values
y_train_onehot = pd.get_dummies(mytrain['Survived']).values

X_test = scaler.transform(mytest[features].values)
y_test = mytest['Survived'].values

l = len(X_test)
X_test = X_test.reshape(l,6)
y_test = y_test.reshape(l,1)

def train_next_batch(i):
    
    sz = X_train.shape[0]
    step = 10
    
    i = i%sz
    n1 = i*step
    n2 = n1 + step
    
    if((n2 >= sz -1) or (n1 >= sz-1)):
        n2 = n2-n1
        n1 = 0
    
    bx = X_train[n1:n2]
    by = y_train[n1:n2]
    
    bx = bx.reshape(step,6)
    by = by.reshape(step,1)
    return bx,by

def train_helper(Y,Yl,X,pkeep=None,numiters=100):
    #correct answers
    print "training, called with numiters: ", numiters
    print "start time: ", datetime.now()
    Y_ = tf.placeholder(tf.float32, [None,1])
    
    lr = tf.placeholder(tf.float32)
    
    alpha = 0.03  
    maxlr = alpha
    minlr = alpha/30.0
    decay_speed = 2000.0
    
    xe = tf.reduce_sum(abs(Yl - Y_))
    train_step = tf.train.AdamOptimizer(lr).minimize(xe)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    Y_int = tf.cast(Y_, tf.int64)
    is_correct = tf.equal(tf.argmax(Y,1),Y_int)
    
    accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
    #accuracy = tf.reduce_mean(is_correct)
    test_data = {X:X_test, Y_:y_test, pkeep: 1.0}
    
    train_next_batch.i = 0
    
    sess.run(tf.global_variables_initializer())
    
    for i in range(numiters):
        lra = minlr + (maxlr - minlr) * math.exp(-i/decay_speed)
        
        bx,by = train_next_batch(i)
        train_data = {X:bx, Y_:by, lr:lra, pkeep: 0.5}
        
        sess.run(train_step,feed_dict=train_data)
        
        if(i%500 == 0):
            print "iter: ", i
            a,c = sess.run([accuracy,xe], feed_dict=test_data)
            print a,c
    
    print "end time: ", datetime.now()
    
    predictions = np.argmax(sess.run(Y, feed_dict={X: X_test}), 1)
    submission = pd.DataFrame({
        "PassengerId": X_test["PassengerId"],
        "Survived": predictions
    })

    submission.to_csv("titanic-submission.csv", index=False)
    return a



#returns weights and biases tensors, given input dims
def wbhelper2(a,b):
    W = tf.Variable(tf.truncated_normal([a,b], stddev=0.1))
    #b = tf.Variable(tf.constant(0.1,shape=[cout]))
    bi = tf.Variable(tf.ones([b])/10)
    #bi = tf.constant(0.1, tf.float32, [b])
    return W,bi

X = tf.placeholder(tf.float32, [None,6])  #none is for later assignment

W1,b1 = wbhelper2(6,32)
W2,b2 = wbhelper2(32,32)
W3,b3 = wbhelper2(32,1)

acf = tf.nn.relu
pkeep = tf.placeholder(tf.float32)
Y1f = acf(tf.matmul(X,W1) + b1)
Y1 = tf.nn.dropout(Y1f, pkeep)
Y2f = acf(tf.matmul(Y1,W2) + b2)
Y2 = tf.nn.dropout(Y2f, pkeep)
Y3l = tf.matmul(Y2,W3) + b3
#Y = tf.nn.softmax(Y3l)
Y = Y3l

print train_helper(Y, Y3l, X, pkeep, numiters=20000)
