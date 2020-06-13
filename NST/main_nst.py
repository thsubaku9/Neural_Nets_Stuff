import numpy as np
import tensorflow as tf
import meta
from minifier import *
import matplotlib.pyplot as plt

#create Initial image
GenImg = tf.Variable(initial_value = tf.random.uniform(shape = (1,meta.shapeImg[0],meta.shapeImg[1],meta.shapeImg[2]), minval = 0, maxval = 1,dtype = tf.float32), dtype = tf.float32, shape = (1,meta.shapeImg[0],meta.shapeImg[1],meta.shapeImg[2]), name = "GeneratedImg")

#create classifier
classifier = miniClassifier(meta.joinedData,meta.labelsOneHot)
classifier.train_init()
classifier.compile(50)

#create content loss
def content_loss(content_activation, generated_activation):
    return tf.reduce_mean(tf.square(content_activation - generated_activation))    

#create style loss
def gram_matrix(convoluted_layer):
    #you can either transpose a or b, but ultimately you're optimizing loss on the sum difference
    channels = convoluted_layer.shape[-1]
    flat = tf.reshape(convoluted_layer,[-1,channels])
    gramMat = tf.matmul(flat,flat,transpose_a = True)    
    return gramMat    

def style_loss(style_layer, generated_layer):     
    _,h,w,c = generated_layer.get_shape().as_list()
    gram_style = gram_matrix(style_layer)
    gram_gen = gram_matrix(generated_layer) 
    return tf.reduce_mean(tf.square(gram_style - gram_gen)/(h*w*c)**2)

#load the frozen network and create graph
nst_w1 = tf.constant(value = classifier.retrieveLayers["w1"], shape = classifier.retrieveLayers["w1"].shape, dtype = tf.float32)
nst_w2 = tf.constant(value = classifier.retrieveLayers["w2"], shape = classifier.retrieveLayers["w2"].shape, dtype = tf.float32)
nst_w3 = tf.constant(value = classifier.retrieveLayers["w3"], shape = classifier.retrieveLayers["w3"].shape, dtype = tf.float32)
nst_b1 = tf.constant(value = classifier.retrieveLayers["b1"], shape = classifier.retrieveLayers["b1"].shape, dtype = tf.float32)
nst_b2 = tf.constant(value = classifier.retrieveLayers["b2"], shape = classifier.retrieveLayers["b2"].shape, dtype = tf.float32)
nst_b3 = tf.constant(value = classifier.retrieveLayers["b3"], shape = classifier.retrieveLayers["b3"].shape, dtype = tf.float32)
fc1w = tf.constant(value = classifier.retrieveLayers["fc1w"], shape = classifier.retrieveLayers["fc1w"].shape, dtype = tf.float32)
fc1b = tf.constant(value = classifier.retrieveLayers["fc1b"], shape = classifier.retrieveLayers["fc1b"].shape, dtype = tf.float32)
fc2w = tf.constant(value = classifier.retrieveLayers["fc2w"], shape = classifier.retrieveLayers["fc2w"].shape, dtype = tf.float32)
fc2b = tf.constant(value = classifier.retrieveLayers["fc2b"], shape = classifier.retrieveLayers["fc2b"].shape, dtype = tf.float32)


a1 = 0.6
a2 = 0.3
a3 = 0.3

#forward_pass

def fullcon_internal(W,b,layer,keep_prob):
    fc = tf.nn.bias_add(tf.matmul(layer,W),b)
    dropped = tf.nn.dropout(fc, keep_prob = keep_prob)
    return dropped

conv1 = classifier.conv_layer(GenImg,nst_w1,nst_b1,'VALID', 'gen_conv1')
pool1 = classifier.avg_pooling(conv1,"gen_pool1") #gram matrix feed

conv2 = classifier.conv_layer(pool1,nst_w2,nst_b2,'VALID', 'gen_conv2')
pool2 = classifier.avg_pooling(conv2,"gen_pool2") #gram matrix feed

conv3 = classifier.conv_layer(pool2,nst_w3,nst_b3,'VALID', 'gen_conv3')
pool3 = classifier.avg_pooling(conv3,"gen_pool3")

pool4 = classifier.max_pooling(pool3,"gen_pool4")
flat = classifier.flatten(pool4)

fc1 = fullcon_internal(fc1w, fc1b, flat, keep_prob = 0.5)
relu1 = tf.nn.relu(fc1)
fc2 = fullcon_internal(fc2w, fc2b, relu1 , keep_prob = 0.8)

#backprop with loss minimization
loss_cost = content_loss(classifier.retrieveLayers['contentpreout'], fc2)*a1 + style_loss(classifier.retrieveLayers['style1'][1], pool1)*a2 + style_loss(classifier.retrieveLayers['style2'][0], pool2)*a3
optimizer = tf.train.ProximalGradientDescentOptimizer(0.007, 0.4).minimize(loss_cost)

#commence iterations
print("Image generation starting\n")
for i in range(100):    
    curr_loss = classifier.sess.run(loss_cost)
    classifier.sess.run(optimizer)
    print(curr_loss)
#done
