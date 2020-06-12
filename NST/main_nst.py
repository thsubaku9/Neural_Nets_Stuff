import numpy as np
import tensorflow as tf
import meta
from minifier import *

#create Initial image
GenImg = tf.Variable(initial_value = tf.random.uniform(shape = meta.shapeImg, minval = 0, maxval = 1,dtype = tf.float32), dtype = tf.float32, shape = meta.shapeImg, name = "GeneratedImg")

#create classifier
classifier = miniClassifier(meta.joinedData,meta.labelsOneHot)
classifier.train_init()
classifier.compile(50)

#create content loss
def content_loss(content_activation, generated_activation):
    tf.reduce_mean(tf.square(content_activation - generated_activation))    

#create style loss
def gram_matrix(convoluted_layer):
    #you can either transpose a or b, but ultimately you're optimizing loss on the sum difference
    channels = convoluted_layer.shape[-1]
    flat = tf.reshape(convoluted_layer,[-1,channels])
    gramMat = tf.matmul(flat,flat,transpose_a = True)    
    return gramMat
    return

def style_loss(style_layer, generated_layer):
    h,w,c = style_layer.get_shape().as_list()
    hg,wg,cg = generated_layer.get_shape().as_list()
    gram_style = gram_matrix(style_layer)
    gram_gen = gram_matrix(generated_layer)
    assert h == hg and w == wg and c == cg
    
    return tf.reduce_mean(tf.square(gram_style - gram_gen)/(h*w*c)**2)

#create total loss and value assignments

#minimize total loss
#done
