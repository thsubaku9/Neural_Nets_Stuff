#currently unfinished

import tensorflow as tf
import meta

tf.compat.v1.reset_default_graph()

class ImageClassifier():
    #vgg19 is used here
    def __init__(self, X, Y, totalClasses = 2, preOutput = 2000,):
        self.learn_rate = 0.001
        self.Img = X
        self.Label = Y
        self.totalClasses = totalClasses
        self.preOutputSize = preOutput

        self.input = tf.placeholder(dtype = tf.float32, shape = (None,self.Img.shape[1],self.Img.shape[2],self.Img.shape[3]))
        self.output = tf.placeholder(dtype = tf.float32, shape = (None,self.Label.shape[1]))

    # add names to layer later to get style and context loss

    def set_learning_rate(self,lr):
        if(lr >0 and lr <1):
            self.learn_rate = lr

            
    def conv_layer(self, inFeed, tfFilter, bias):
        #VALID -> ensures shrinking
        #SAME -> ensures same dimension
        convolve = tf.nn.conv2d(inFeed ,tfFilter, strides = (1,1,1,1), padding = 'VALID')
        biased = tf.nn.bias_add(convolve, bias)
        return tf.nn.leaky_relu(biased, alpha = 0.3)
    
    def max_pooling(self,inFeed,name):
        return tf.nn.max_pool(inFeed, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME", name = name)
    
    def avg_pooling(self,inFeed,name):
        return tf.nn.avg_pool(inFeed, ksize = [1,2,2,1], strides = [1,2,2,1] ,padding = "SAME", name = name)
        
    def flatten(self, layer):
        
        ln = len(layer.shape)
        totalNeurons = 1
        for i in range(1,ln):
            totalNeurons*=layer.shape[i].value
        return tf.reshape(layer,[-1,totalNeurons])
    
    def fullcon(self, layer, name, inputNeurons, outputNeurons):
        W = tf.Variable(name = name, shape = (inputNeurons,outputNeurons), dtype = tf.float32, initial_value = tf.random.normal((inputNeurons,outputNeurons)))
        b = tf.Variable(name = name, shape = (outputNeurons,), dtype = tf.float32, initial_value = tf.random.normal((outputNeurons,)))

        self.weights[name] = W
        self.biases[name] = b
        
        fc = tf.nn.bias_add(tf.matmul(layer,W),b)
        
        return fc
        
    def build(self):

        # image has 3 layers, first output is 64 layers
        # second input has 64 layers, second output is 64 layers
        # so on and so forth
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

        self.weights = weights; self.biases = biases;
        
        self.conv1_1 = self.conv_layer(self.input, weights["w1_1"], biases["b1_1"])
        self.conv1_2 = self.conv_layer(self.conv1_1, weights["w1_2"], biases["b1_2"])
        self.pool1 = self.max_pooling(self.conv1_2,"max_pool1")

        self.conv2_1 = self.conv_layer(self.pool1, weights["w2_1"], biases["b2_1"])
        self.conv2_2 = self.conv_layer(self.conv2_1, weights["w2_2"], biases["b2_2"])
        self.pool2 = self.max_pooling(self.conv2_2,"max_pool2")

        self.conv3_1 = self.conv_layer(self.pool2, weights["w3_1"], biases["b3_1"]) 
        self.conv3_2 = self.conv_layer(self.conv3_1, weights["w3_2"], biases["b3_2"])
        self.conv3_3 = self.conv_layer(self.conv3_2, weights["w3_3"], biases["b3_3"])
        self.conv3_4 = self.conv_layer(self.conv3_3, weights["w3_4"], biases["b3_4"])
        self.pool3 = self.max_pooling(self.conv3_4,"max_pool3")
        
        self.conv4_1 = self.conv_layer(self.pool3, weights["w4_1"], biases["b4_1"])
        self.conv4_2 = self.conv_layer(self.conv4_1, weights["w4_2"], biases["b4_2"])
        self.conv4_3 = self.conv_layer(self.conv4_2, weights["w4_3"], biases["b4_3"])
        self.conv4_4 = self.conv_layer(self.conv4_3, weights["w4_4"], biases["b4_4"])
        self.pool4 = self.avg_pooling(self.conv4_4,"avg_pool1")

        self.conv5_1 = self.conv_layer(self.pool4, weights["w5_1"], biases["b5_1"])
        self.conv5_2 = self.conv_layer(self.conv5_1, weights["w5_2"], biases["b5_2"])
        self.conv5_3 = self.conv_layer(self.conv5_2, weights["w5_3"], biases["b5_3"])
        self.conv5_4 = self.conv_layer(self.conv5_3, weights["w5_4"], biases["b5_4"])
        self.pool5 = self.avg_pooling(self.conv5_4,"avg_pool2")

        self.flat = self.flatten(self.pool5)        
                
        self.fc1 = self.fullcon(self.flat,"fc1",self.flat.shape[1].value,self.preOutputSize)
        self.relu1 = tf.nn.relu(self.fc1)

        self.fc2 = self.fullcon(self.relu1,"fc2",self.preOutputSize,self.totalClasses)
        self.classifierOutput = tf.nn.softmax(self.fc2,name = "result")
        
        return self.classifierOutput

    def optimize(self,logits):
        
        classifier_acc = tf.nn.softmax_cross_entropy_with_logits(labels = self.output, logits = logits)
        loss = tf.reduce_mean(classifier_acc)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learn_rate).minimize(loss)
        

    def train(self):

        init = tf.initialize_all_variables()
        sess = tf.Session()            
        
'''
with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()
'''
