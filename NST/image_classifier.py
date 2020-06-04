#currently unfinished

import tensorflow as tf
#import meta

tf.compat.v1.reset_default_graph()

#init = tf.initialize_all_variables()

sess = tf.Session()

class ImageClassifier():
    #vgg19 is used here
    def __init__(self,X,Y):
        self.Img = X
        self.Label = Y
            
    def conv_layer(self, inFeed, tfFilter, bias):
        convolve = tf.nn.conv2d(inFeedm tfFilter, strides = (1,1,1,1), padding = 'SAME')
        biased = tf.nn.bias_add(convolve, bias)
        return tf.nn.leaky_relu(biased, alpha = 0.3)
    
    def max_pooling(self,inFeed,name):
        return tf.nn.max_pool(inFeed, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME", name = name)
    
    def avg_pooling(self,inFeed,name):
        return tf.nn.avg_pool(inFeed, ksize = [1,2,2,1], strides = [1,2,2,1] ,padding = "SAME", name = name)
        
    def build(self):

        weights = {}
        
        self.conv1_1 = self.conv_layer(self.Img, weights[""], weights[""])
        self.conv1_2 = self.conv_layer(self.conv1_1, weights[""], weights[""])
        self.pool1 = self.avg_pooling(self.conv1_2,"max_pool1")

        self.conv2_1 = self.conv_layer(self.pool1, , weights[""], weights[""])
        self.conv2_2 = self.conv_layer(self.conv2_1, , weights[""], weights[""])
        self.pool2 = 
