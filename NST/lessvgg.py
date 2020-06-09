#classless vgg19

import tensorflow as tf
import meta

tf.compat.v1.reset_default_graph()

learn_rate = 0.001

Img = meta.joinedData
Label = meta.labelsOneHot
totalClasses = 2
preOutputSize = 2000

input = tf.compat.v1.placeholder(dtype = tf.float32, shape = (None,Img.shape[1],Img.shape[2],Img.shape[3]))
output = tf.compat.v1.placeholder(dtype = tf.float32, shape = (None,Label.shape[1]))


weights = {
            'w1_1': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,3,64]),dtype = tf.float32,shape = [3,3,3,64], name = 'w1_1'),
            'w1_2': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,64,64]),dtype = tf.float32,shape = [3,3,64,64], name = 'w1_2'),
            'w2_1': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,64,128]),dtype = tf.float32,shape = [3,3,64,128], name = 'w2_1'),
            'w2_2': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,128,128]),dtype = tf.float32,shape = [3,3,128,128], name = 'w2_2'),
            'w3_1': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,128,256]),dtype = tf.float32,shape = [3,3,128,256], name = 'w3_1'),
            'w3_2': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,256,256]),dtype = tf.float32,shape = [3,3,256,256], name = 'w3_2'),
            'w3_3': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,256,256]),dtype = tf.float32,shape = [3,3,256,256], name = 'w3_3'),
            'w3_4': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,256,256]),dtype = tf.float32,shape = [3,3,256,256], name = 'w3_4'),
            'w4_1': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,256,512]),dtype = tf.float32,shape = [3,3,256,512], name = 'w4_1'),
            'w4_2': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,512,512]),dtype = tf.float32,shape = [3,3,512,512], name = 'w4_2'),
            'w4_3': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,512,512]),dtype = tf.float32,shape = [3,3,512,512], name = 'w4_3'),
            'w4_4': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,512,512]),dtype = tf.float32,shape = [3,3,512,512], name = 'w4_4'),
            'w5_1': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,512,512]),dtype = tf.float32,shape = [3,3,512,512], name = 'w5_1'),
            'w5_2': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,512,512]),dtype = tf.float32,shape = [3,3,512,512], name = 'w5_2'),
            'w5_3': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,512,512]),dtype = tf.float32,shape = [3,3,512,512], name = 'w5_3'),
            'w5_4': tf.Variable(initial_value = tf.contrib.layers.xavier_initializer(seed=0)([3,3,512,512]),dtype = tf.float32,shape = [3,3,512,512], name = 'w5_4'),        
            }

biases = {    
            'b1_1': tf.Variable(initial_value = tf.random.normal(shape = (64,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (64,), name = 'b1_1'),
            'b1_2': tf.Variable(initial_value = tf.random.normal(shape = (64,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (64,), name = 'b1_2'),
            'b2_1': tf.Variable(initial_value = tf.random.normal(shape = (128,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (128,), name = 'b2_1'),
            'b2_2': tf.Variable(initial_value = tf.random.normal(shape = (128,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (128,), name = 'b2_2'),
            'b3_1': tf.Variable(initial_value = tf.random.normal(shape = (256,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (256,), name = 'b3_1'),
            'b3_2': tf.Variable(initial_value = tf.random.normal(shape = (256,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (256,), name = 'b3_2'),
            'b3_3': tf.Variable(initial_value = tf.random.normal(shape = (256,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (256,), name = 'b3_3'),
            'b3_4': tf.Variable(initial_value = tf.random.normal(shape = (256,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (256,), name = 'b3_4'),
            'b4_1': tf.Variable(initial_value = tf.random.normal(shape = (512,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (512,), name = 'b4_1'),
            'b4_2': tf.Variable(initial_value = tf.random.normal(shape = (512,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (512,), name = 'b4_2'),
            'b4_3': tf.Variable(initial_value = tf.random.normal(shape = (512,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (512,), name = 'b4_3'),
            'b4_4': tf.Variable(initial_value = tf.random.normal(shape = (512,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (512,), name = 'b4_4'),
            'b5_1': tf.Variable(initial_value = tf.random.normal(shape = (512,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (512,), name = 'b5_1'),
            'b5_2': tf.Variable(initial_value = tf.random.normal(shape = (512,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (512,), name = 'b5_2'),
            'b5_3': tf.Variable(initial_value = tf.random.normal(shape = (512,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (512,), name = 'b5_3'),
            'b5_4': tf.Variable(initial_value = tf.random.normal(shape = (512,), mean = 0.0, stddev = 1.5), dtype = tf.float32,shape = (512,), name = 'b5_4'),
            }

def set_learning_rate(lr):
    if(lr >0 and lr <1):
        learn_rate = lr
    
def conv_layer( inFeed, tfFilter, bias):
    #VALID -> ensures shrinking
    #SAME -> ensures same dimension
    convolve = tf.nn.conv2d(inFeed ,tfFilter, strides = (1,1,1,1), padding = 'VALID')
    biased = tf.nn.bias_add(convolve, bias)
    return tf.nn.leaky_relu(biased, alpha = 0.3)

def max_pooling(inFeed,name):
    return tf.nn.max_pool(inFeed, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME", name = name)

def avg_pooling(inFeed,name):
    return tf.nn.avg_pool(inFeed, ksize = [1,2,2,1], strides = [1,2,2,1] ,padding = "SAME", name = name)
	
def flatten( layer):
    ln = len(layer.shape)
    totalNeurons = 1
    for i in range(1,ln):        
        totalNeurons*=layer.shape[i].value
    return tf.reshape(layer,[-1,totalNeurons])

def fullcon( layer, name, inputNeurons, outputNeurons):
    W = tf.Variable(name = name, shape = (inputNeurons,outputNeurons), dtype = tf.float32, initial_value = tf.random.normal((inputNeurons,outputNeurons)))
    b = tf.Variable(name = name, shape = (outputNeurons,), dtype = tf.float32, initial_value = tf.random.normal((outputNeurons,)))
    weights[name] = W
    biases[name] = b	
    fc = tf.nn.bias_add(tf.matmul(layer,W),b)
    return fc

def build():
    conv1_1 = conv_layer(input, weights["w1_1"], biases["b1_1"])
    conv1_2 = conv_layer(conv1_1, weights["w1_2"], biases["b1_2"])
    pool1 = max_pooling(conv1_2,"max_pool1")
    
    conv2_1 = conv_layer(pool1, weights["w2_1"], biases["b2_1"])
    conv2_2 = conv_layer(conv2_1, weights["w2_2"], biases["b2_2"])
    pool2 = max_pooling(conv2_2,"max_pool2")
    
    conv3_1 = conv_layer(pool2, weights["w3_1"], biases["b3_1"]) 
    conv3_2 = conv_layer(conv3_1, weights["w3_2"], biases["b3_2"])
    conv3_3 = conv_layer(conv3_2, weights["w3_3"], biases["b3_3"])
    conv3_4 = conv_layer(conv3_3, weights["w3_4"], biases["b3_4"])
    pool3 = max_pooling(conv3_4,"max_pool3")
    
    conv4_1 = conv_layer(pool3, weights["w4_1"], biases["b4_1"])
    conv4_2 = conv_layer(conv4_1, weights["w4_2"], biases["b4_2"])
    conv4_3 = conv_layer(conv4_2, weights["w4_3"], biases["b4_3"])
    conv4_4 = conv_layer(conv4_3, weights["w4_4"], biases["b4_4"])
    pool4 = avg_pooling(conv4_4,"avg_pool1")
    
    conv5_1 = conv_layer(pool4, weights["w5_1"], biases["b5_1"])
    conv5_2 = conv_layer(conv5_1, weights["w5_2"], biases["b5_2"])
    conv5_3 = conv_layer(conv5_2, weights["w5_3"], biases["b5_3"])
    conv5_4 = conv_layer(conv5_3, weights["w5_4"], biases["b5_4"])
    pool5 = avg_pooling(conv5_4,"avg_pool2")
    
    flat = flatten(pool5)        
    fc1 = fullcon(flat,"fc1",flat.shape[1].value,preOutputSize)
    relu1 = tf.nn.relu(fc1)
    fc2 = fullcon(relu1,"fc2",preOutputSize,totalClasses)
    classifierOutput = tf.nn.softmax(fc2,name = "result")

    return classifierOutput

def optimize(logits):
    entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels = output, logits = logits)
    cost = tf.reduce_mean(entropy_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(output,1))
    acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    return cost,optimizer,acc



'''

    def train(iters = 100,batches = None):

        classifier = build()
        cost,optimizer,accuracy = optimize(classifier)
        
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            print("Starting Session\n")
            
            sess.run(init)            
            for it in range(iters):
                if(batches == None):
                    batch_x = Img
                    batch_y = Label
                    run_compute(sess,batch_x,batch_y,cost,optimizer,accuracy)                    
                else:
                    for batch_pointer in range(len(Img)//batches):                        
                        batch_x = Img[batch_pointer*batches:min((batch_pointer+1)*batches,len(Img))]
                        batch_y = Label[batch_pointer*batches:min((batch_pointer+1)*batches,len(Label))]
                        run_compute(sess,batch_x,batch_y,cost,optimizer,accuracy)
                    
    def run_compute(sess,feed_x,feed_y,cost,optimizer,accuracy):
        opt = sess.run(optimizer,feed_dict={x: feed_x, y: feed_y})
        loss,acc = sess.run([cost,accuracy],feed_dict={x: feed_x, y: feed_y})
        print("Loss= {:.5f} , Training Acc = {:.5f}".format(loss,acc))

'''
