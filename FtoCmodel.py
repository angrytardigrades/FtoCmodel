#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is python code and you can use google colab to test this
#==========================
import tensorflow as tf
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
#==========================
#Data
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

#layer 0 for CNN
  l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# the model: simple linear model from Keras API
  model = tf.keras.Sequential([l0])

  model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])
  

  model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
  
  history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")


import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

# put celsius here and see the result and see how the model predict the Fahrenheit value.
# i know converting these two has clear formula , and instead of this model you can simply use that formula :) but the purpose of writing this code is to show how ML can be used.
print(model.predict([100.0]))
