import numpy as np
import tensorflow as tf
import meta
from minifier import *
import matplotlib.pyplot as plt

#create classifier
tf.reset_default_graph()
classifier = miniClassifier(meta.joinedData,meta.labelsOneHot,totalClasses = 3)
classifier.train_init()
classifier.compile(120)
classifier.sess.close()

#new session graph to ensure you don't run out of resources
g = tf.Graph()
g.as_default(); g.device('/gpu:0');
tf.reset_default_graph()
#GPU fix
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
#GPU fix

#create image loss
def image_loss(Img,base_img):    
    return tf.reduce_sum(tf.abs(tf.subtract(Img[0],base_img)))
    
#create content loss
def content_loss(content_activation, generated_activation):
    return tf.reduce_mean(tf.square(tf.subtract(content_activation,generated_activation)/generated_activation.shape[1].value))

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
    return tf.reduce_mean(tf.abs(tf.square(gram_style,gram_gen))/((h*w*c)**2))

#create Initial image
GenImg = tf.Variable(initial_value = tf.random.uniform(shape = (1,meta.shapeImg[0],meta.shapeImg[1],meta.shapeImg[2]), minval = 0, maxval = 255,dtype = tf.float32), dtype = tf.float32, shape = (1,meta.shapeImg[0],meta.shapeImg[1],meta.shapeImg[2]), name = "GeneratedImg")

#load the frozen network and create graph
nst_w1_1 = tf.constant(value = classifier.retrieveLayers["w1_1"], shape = classifier.retrieveLayers["w1_1"].shape, dtype = tf.float32)
nst_w1_2 = tf.constant(value = classifier.retrieveLayers["w1_2"], shape = classifier.retrieveLayers["w1_2"].shape, dtype = tf.float32)
nst_w2_1 = tf.constant(value = classifier.retrieveLayers["w2_1"], shape = classifier.retrieveLayers["w2_1"].shape, dtype = tf.float32)
nst_w2_2 = tf.constant(value = classifier.retrieveLayers["w2_2"], shape = classifier.retrieveLayers["w2_2"].shape, dtype = tf.float32)
nst_w3 = tf.constant(value = classifier.retrieveLayers["w3"], shape = classifier.retrieveLayers["w3"].shape, dtype = tf.float32)
nst_b1_1 = tf.constant(value = classifier.retrieveLayers["b1_1"], shape = classifier.retrieveLayers["b1_1"].shape, dtype = tf.float32)
nst_b1_2 = tf.constant(value = classifier.retrieveLayers["b1_2"], shape = classifier.retrieveLayers["b1_2"].shape, dtype = tf.float32)
nst_b2_1 = tf.constant(value = classifier.retrieveLayers["b2_1"], shape = classifier.retrieveLayers["b2_1"].shape, dtype = tf.float32)
nst_b2_2 = tf.constant(value = classifier.retrieveLayers["b2_2"], shape = classifier.retrieveLayers["b2_2"].shape, dtype = tf.float32)
nst_b3 = tf.constant(value = classifier.retrieveLayers["b3"], shape = classifier.retrieveLayers["b3"].shape, dtype = tf.float32)
fc1w = tf.constant(value = classifier.retrieveLayers["fc1w"], shape = classifier.retrieveLayers["fc1w"].shape, dtype = tf.float32)
fc1b = tf.constant(value = classifier.retrieveLayers["fc1b"], shape = classifier.retrieveLayers["fc1b"].shape, dtype = tf.float32)
fc2w = tf.constant(value = classifier.retrieveLayers["fc2w"], shape = classifier.retrieveLayers["fc2w"].shape, dtype = tf.float32)
fc2b = tf.constant(value = classifier.retrieveLayers["fc2b"], shape = classifier.retrieveLayers["fc2b"].shape, dtype = tf.float32)

total_iterations = 300
content_weight = 0.8
style_weight = 0.5
imagedelta_weight = 0.2
learning_rate = 20.0
lost_penalty = 0.001
#helper functions

def fullcon_internal(W,b,layer,keep_prob):
    fc = tf.nn.bias_add(tf.matmul(layer,W),b)
    dropped = tf.nn.dropout(fc, keep_prob = keep_prob)
    return dropped

#NST-Graph


with tf.Session(config = config) as sess:
    #forward_pass
    conv1_1 = classifier.conv_layer(GenImg,nst_w1_1,nst_b1_1,'VALID', 'gen_conv1_1')
    conv1_2 = classifier.conv_layer(conv1_1,nst_w1_2,nst_b1_2,'VALID', 'gen_conv1_2')
    pool1 = classifier.avg_pooling(conv1_2,"gen_pool1") #gram matrix feed

    conv2_1 = classifier.conv_layer(pool1,nst_w2_1,nst_b2_1,'VALID', 'gen_conv2_1')
    conv2_2 = classifier.conv_layer(conv2_1,nst_w2_2,nst_b2_2,'VALID', 'gen_conv2_2')
    pool2 = classifier.max_pooling(conv2_2,"gen_pool2") #gram matrix feed

    conv3 = classifier.conv_layer(pool2,nst_w3,nst_b3,'VALID', 'gen_conv3')
    pool3 = classifier.avg_pooling(conv3,"gen_pool3")
    
    flat = classifier.flatten(pool3)

    fc1 = fullcon_internal(fc1w, fc1b, flat, keep_prob = 0.7)
    relu1 = tf.nn.relu(fc1)
    fc2 = fullcon_internal(fc2w, fc2b, relu1 , keep_prob = 0.8)

    #backprop with loss minimization
    #0 index -> style
    #1 index -> content
    content_cost =  content_loss(classifier.retrieveLayers['contentprepreout'][1], relu1) #+ content_loss(classifier.retrieveLayers['contentpreout'][1], fc2) 
    style_cost = style_loss(classifier.retrieveLayers['style1'][0], pool1) + style_loss(classifier.retrieveLayers['style2'][0], pool2) #+ style_loss(classifier.retrieveLayers['style3'][0],pool3)
    loss_cost = lost_penalty*(image_loss(GenImg,meta.joinedData[1])*imagedelta_weight + content_cost*content_weight + style_cost*style_weight)
    optimizer = tf.train.AdamOptimizer(learning_rate,0.9,0.999,1e-08).minimize(loss_cost, var_list =[GenImg])
    init = tf.global_variables_initializer()
    sess.run(init)
    #commence iterations
    print("Commencing Neural Style Transfer\n")
    for i in range(total_iterations):
        #add independent loss section
        curr_loss = sess.run(loss_cost)
        sess.run(optimizer)
        clipper = tf.clip_by_value(GenImg,0.0,255.0)
        GenImg.assign(clipper)
        img = sess.run(clipper)
        if(i%10 == 0 and i>0):
            #add an exp gain for learning rate ?
            print(curr_loss)
            learning_rate *= 2.1
            lost_penalty *= 5 
            plt.imshow(np.int16(img[0]))
            plt.show()
            

#done
