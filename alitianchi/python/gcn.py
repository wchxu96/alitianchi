from keras import activations,initializers,constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K

# a very naive gcn implement that the input is a matrix of the wordvector(node(230),wordvec(300)) and a
# weighted adjacent matrix of the graph we construct before(i,e a list of tensors) and output the feature
# matrix where each line is a 2048 dim vector as a recognizer of each class. the id is hacky.
class gcn(Layer):
    def __init__(self, units, activation='relu', use_bias=True,kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constriant=None,**kwargs):
        super(gcn, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activations)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(constraints)
        self.bias_constraint = bias_constriant

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        features = inputs[0]
        adjacent_matrix = inputs[1]
        output_mid = K.dot(adjacent_matrix,features)
        output = K.dot(output_mid,self.kernel)
        if self.use_bias:
            output += self.bias
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0] # wordvec
        output_shape = (features_shape[0],self.units)
        return output_shape

    def build(self, input_shape):
        features_shape = input_shape[0]
        wordvec_dim = features_shape[1] # 300
        self.kernel = self.add_weight(shape=(wordvec_dim,self.units),initializer=self.kernel_initializer,
                                      name='kernel',regularizer=self.kernel_initializer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),initializer=self.bias_initializer,name='bias',
                                        regularizer=self.bias_initializer,constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True
