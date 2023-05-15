 <h2> Regression With Multi Layer Perceptron and Tensorflow</h2>
 
 
 <p> In order to estimate a value using Tensowrflow by Python you need to import the following Packages. </p>
 <code>
 from sklearn.datasets import fetch_california_housing
 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import StandardScaler
 import tensorflow as tf
 from tensorflow import keras
 import pandas as pd
 import matplotlib.pyplot as plt
 </code>
 
 <H3> Dataset </H3>
 
 <p> After that we must include our dataset. In this project we have used California Housing Prices dataset containing these features :  </p>
 
 <ul>
  <li>Longitude</li>
     <li>latitude</li>
     <li>The age of the building</li>
     <li>number of rooms</li>
     <li>Number of bedrooms</li>
     <li>Mitigation of pollution in the area</li>
     <li>The owner of the houses</li>
     <li>Income</li>
     <li>building price</li>
     <li>Close to the ocean</li>
 <ul>
  
<code> housing = fetch_california_housing()
</code>
<p> Then we need to organise our train,validation and test sets. Here in the following we have prepared it: </p>

  
<code>
  x_train_full, x_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
  x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full)
 </code>
 
 <p> If we print our data,we would know that they are not normal so we used codes below for standardisation of our data.  </p>
 
 
  
 <code> 
  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_valid = scaler.transform(x_valid)
  x_test = scaler.transform(x_test) 
 </code>
  
 <h3> Model </h3>
  <p> For Training we need a model. With keral we have used sequentional mode for our aim. </p>
  
<code>
 model = keras.models.Sequential([
keras.layers.Dense(30, activation="relu", input_shape=x_train.shape[1:]),
keras.layers.Dense(1)
])
</code>
<p> The next step is building our layers. The first layer is formed based on our input. Our Input data is an image which has 28*28 dimenstion. As a consequence our code in python would be:  </p>

<code> 
  model.add(keras.layers.Flatten(input_shape=[28, 28]))
</code>
 
 <p> For selecting your model you have to consider this issue that it takes a while to be proficient in selecting your proper model.</p>
 
<code>
 model = keras.models.Sequential([
 keras.layers.Dense(30, activation="relu", input_shape=x_train.shape[1:]),
 keras.layers.Dense(1)
 ])
</code>
 
 <p>
 In addtion, before fitting you can view a summery of your model.
 </p>
 <code>
  model.compile(loss="mean_squared_error", optimizer="sgd")
 </code>
 <h3> Fit and Evaluate </h3>
 <p>
 The last step is compiling and fiting our model. Pure and Simple
 </p>
 <code>
   model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
   history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))
 </code>

<p> 
 For showing every epoch result we can use matplotlib which is available in python and at the beginning we have imported it in our project. Here we have our code using data on <i> history varaible </i> for illustrating.
 </p>
 
 <code>
  pd.DataFrame(history.history).plot(figsize=(8, 5))
  plt.grid(True)
  plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
  plt.show()
</code>
 <img src='https://github.com/AIAML/Regression_using_multi_layer_perceptron/blob/master/myplot.png' />
