from keras.layers import Layer
from keras import backend as K

class Attention(Layer):
    def __init__(self, neurons, **kwargs):
        self.neurons = neurons
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], self.neurons), 
                               initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(self.neurons,), 
                               initializer='zeros', trainable=True)        
        super(Attention, self).build(input_shape)
 
    def call(self, x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x, self.W) + self.b)
        # Compute the weights
        alpha = K.softmax(e)
        # Compute the context vector
        context = x * alpha
        return context

    def compute_output_shape(self, input_shape):
        return input_shape

