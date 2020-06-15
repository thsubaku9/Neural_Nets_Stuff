#miniClassifier (for poor mans laptop)

import tensorflow as tf
import numpy as np

tf.compat.v1.reset_default_graph()

class miniClassifier():
    def __init__(self, X, Y, totalClasses = 2, preOutput = 2000):
        self.learn_rate = 0.003
        self.Img = X
        self.Label = Y
        self.totalClasses = totalClasses
        self.preOutputSize = preOutput
        
        self.save_path = None
        self.sess = None
        self.retrieveLayers = {}
        
        self.input = tf.compat.v1.placeholder(dtype = tf.float32, shape = (None,self.Img.shape[1],self.Img.shape[2],self.Img.shape[3]))
        self.output = tf.compat.v1.placeholder(dtype = tf.float32, shape = (None,self.Label.shape[1]))

        #q = tf.Graph()
        #q.as_default(); q.device('/cpu:0');
    def set_learning_rate(self,lr):
        if(lr >0 and lr <1):
            self.learn_rate = lr
            
    def conv_layer(self, inFeed, tfFilter, bias,padding_type, name):
        #VALID -> ensures shrinking
        #SAME -> ensures same dimension
        convolve = tf.nn.conv2d(inFeed ,tfFilter, strides = (1,1,1,1), padding = padding_type)
        biased = tf.nn.bias_add(convolve, bias)
        return tf.nn.relu(biased,name = name)
        #return tf.nn.leaky_relu(biased, alpha = 0.3, name = name)
    
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
    
    def fullcon(self, layer, name, inputNeurons, outputNeurons, keep_prob = 0.3 ):
        W = tf.Variable(name = name, shape = (inputNeurons,outputNeurons), dtype = tf.float32, initial_value = tf.random.normal((inputNeurons,outputNeurons), mean = 0.0, stddev = 1, dtype = tf.float32))
        b = tf.Variable(name = name, shape = (outputNeurons,), dtype = tf.float32, initial_value = tf.random.normal((outputNeurons,), mean = 0.0, stddev = 1, dtype = tf.float32))

        self.weights[name] = W
        self.biases[name] = b        
        fc = tf.nn.bias_add(tf.matmul(layer,W),b)
        dropped = tf.nn.dropout(fc, keep_prob = keep_prob, name = name)
        return dropped

    def build(self):        
        weights = {
            'w1_1': tf.Variable(initial_value = tf.random.uniform(minval = 2, maxval = -2, shape = [5,5,3,6], dtype = tf.float32),dtype = tf.float32,shape = [5,5,3,6], name = 'w1_1'),
            'w1_2': tf.Variable(initial_value = tf.random.uniform(minval = 2, maxval = -2, shape = [5,5,6,6], dtype = tf.float32),dtype = tf.float32,shape = [5,5,6,6], name = 'w1_2'),
            'w2_1': tf.Variable(initial_value = tf.random.uniform(minval = 2, maxval = -2, shape = [5,5,6,12], dtype = tf.float32),dtype = tf.float32,shape = [5,5,6,12], name = 'w2_1'),
            'w2_2': tf.Variable(initial_value = tf.random.uniform(minval = 2, maxval = -2, shape = [7,7,12,12], dtype = tf.float32),dtype = tf.float32,shape = [7,7,12,12], name = 'w2_2'),
            'w3_1': tf.Variable(initial_value = tf.random.uniform(minval = 2, maxval = -2, shape = [5,5,12,8], dtype = tf.float32),dtype = tf.float32,shape = [5,5,12,8], name = 'w3_1'),
            }

        biases = {
            'b1_1': tf.Variable(initial_value = tf.random.normal(shape = (6,), mean = 0.0, stddev = 0.8, dtype = tf.float32), dtype = tf.float32,shape = (6,), name = 'b1_1'),
            'b1_2': tf.Variable(initial_value = tf.random.normal(shape = (6,), mean = 0.0, stddev = 0.8, dtype = tf.float32), dtype = tf.float32,shape = (6,), name = 'b1_2'),
            'b2_1': tf.Variable(initial_value = tf.random.normal(shape = (12,), mean = 0.0, stddev = 0.8, dtype = tf.float32), dtype = tf.float32,shape = (12,), name = 'b2_1'),
            'b2_2': tf.Variable(initial_value = tf.random.normal(shape = (12,), mean = 0.0, stddev = 0.8, dtype = tf.float32), dtype = tf.float32,shape = (12,), name = 'b2_2'),
            'b3_1': tf.Variable(initial_value = tf.random.normal(shape = (8,), mean = 0.0, stddev = 0.8, dtype = tf.float32), dtype = tf.float32,shape = (8,), name = 'b3_1'),            
            }

        self.weights = weights; self.biases = biases;

        self.conv1_1 = self.conv_layer(self.input,weights["w1_1"],biases["b1_1"],'VALID', 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1,weights["w1_2"],biases["b1_2"],'VALID', 'conv1_2')
        self.pool1 = self.avg_pooling(self.conv1_2,"avg_pool1")

        self.conv2_1 = self.conv_layer(self.pool1, weights["w2_1"], biases["b2_1"],'VALID', 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1,weights["w2_2"],biases["b2_2"],'VALID', 'conv2_2')
        self.pool2 = self.max_pooling(self.conv2_2,"avg_pool2")

        self.conv3_1 = self.conv_layer(self.pool2, weights["w3_1"], biases["b3_1"],'VALID', 'conv3_1')        
        self.pool3 = self.avg_pooling(self.conv3_1,"max_pool3")        
        
        self.flat = self.flatten(self.pool3)
        self.fc1 = self.fullcon(self.flat,"fc1",self.flat.shape[1].value,self.preOutputSize,keep_prob = 0.7)
        self.relu1 = tf.nn.relu(self.fc1)

        self.fc2 = self.fullcon(self.relu1,"fc2",self.preOutputSize,self.preOutputSize//2 , keep_prob = 0.8)
        self.sig1 = tf.nn.sigmoid(self.fc2 , name = "content_layer")
        self.fc3 = self.fullcon(self.sig1,"fc3",self.preOutputSize//2,self.totalClasses,keep_prob = 1.0)
        self.classifierOutput = tf.nn.softmax(self.fc3,name = "result")

        return self.classifierOutput

    def optimize(self,logits):        
        entropy_loss = tf.clip_by_value(tf.nn.softmax_cross_entropy_with_logits(labels = self.output, logits = logits),clip_value_min = -1000, clip_value_max = 1000)
        cost = tf.reduce_mean(entropy_loss) + 0.03*tf.nn.l2_loss(self.weights["fc2"]) + 0.02*tf.nn.l2_loss(self.weights["fc1"])
        #optimizer = tf.train.AdamOptimizer(learning_rate = self.learn_rate, beta1 = 0.4 ,beta2 = 0.99).minimize(cost)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate = self.learn_rate).minimize(cost)
        optimizer  = tf.train.GradientDescentOptimizer(learning_rate = self.learn_rate).minimize(cost)
                
        correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(self.output,1))
        acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

        return cost,optimizer,acc

    def train_init(self):
        #run train_init before moving over to compile
        self.classifier = self.build()
        self.cost, self.optimizer, self.accuracy = self.optimize(self.classifier)        
        init = tf.global_variables_initializer()

        #gpu freezes up if either power source is low or memory is unavailble, to avoid the later the below portion has been added
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        #self.config.gpu_options.per_process_gpu_memory_fraction = 0.9
        #GPU-fix-done
        self.sess = tf.Session(config=self.config)
        self.sess.run(init)
        print("Initialization done\n")
        
    def compile(self,iters = 150,batches = None):
        startingAcc = 0
        currentAcc = 0        
        for it in range(iters):
            if(batches == None):
                batch_x = self.Img
                batch_y = self.Label
                currentAcc = self.run_compute(batch_x,batch_y)                
            else:
                intermid_sum = 0
                for batch_pointer in range(len(self.Img)//batches):                    
                    batch_x = self.Img[batch_pointer*batches:min((batch_pointer+1)*batches,len(self.Img))]
                    batch_y = self.Label[batch_pointer*batches:min((batch_pointer+1)*batches,len(self.Label))]
                    intermid_sum += self.run_compute(batch_x,batch_y)
                currentAcc /= batches
            update = self.oneshot_save(startingAcc, currentAcc)    
            if(update):
                startingAcc = min(startingAcc,currentAcc)                
            if(it % 20 == 0):
                    self.learn_rate = self.learn_rate / (1+ 0.05*it)
                    
    def run_compute(self,feed_x,feed_y):
        opt = self.sess.run(self.optimizer,feed_dict={self.input: feed_x, self.output: feed_y})
        loss,acc = self.sess.run([self.cost, self.accuracy],feed_dict={self.input: feed_x, self.output: feed_y})
        print("Loss= {:.5f} , Training Acc = {:.5f}".format(loss,acc))
        return loss #acc

    def oneshot_save(self, startingAcc, currentAcc):
        if (startingAcc < currentAcc) :
            #print("current: {} lowest: {}\n".format(currentLoss, lowestLoss))
            
            self.retrieveLayers['style1'] = self.sess.run(self.pool1, feed_dict={self.input: self.Img})
            self.retrieveLayers['style2'] = self.sess.run(self.pool2, feed_dict={self.input: self.Img })
            self.retrieveLayers['style3'] =self.sess.run(self.pool3, feed_dict={self.input: self.Img })
            self.retrieveLayers['contentprepreout'] = self.sess.run(self.relu1, feed_dict={self.input: self.Img}) 
            self.retrieveLayers['contentpreout'] = self.sess.run(self.fc2, feed_dict={self.input: self.Img}) 
            self.retrieveLayers['w1_1'] = self.sess.run(self.weights['w1_1'])
            self.retrieveLayers['w1_2'] = self.sess.run(self.weights['w1_2'])
            self.retrieveLayers['w2_1'] = self.sess.run(self.weights['w2_1'])
            self.retrieveLayers['w2_2'] = self.sess.run(self.weights['w2_2'])
            self.retrieveLayers['w3'] = self.sess.run(self.weights['w3_1'])
            self.retrieveLayers['b1_1'] = self.sess.run(self.biases['b1_1'])
            self.retrieveLayers['b1_2'] = self.sess.run(self.biases['b1_2'])
            self.retrieveLayers['b2_1'] = self.sess.run(self.biases['b2_1'])
            self.retrieveLayers['b2_2'] = self.sess.run(self.biases['b2_2'])
            self.retrieveLayers['b3'] = self.sess.run(self.biases['b3_1'])
            self.retrieveLayers['fc1w'] = self.sess.run(self.weights['fc1'])
            self.retrieveLayers['fc1b'] = self.sess.run(self.biases['fc1'])
            self.retrieveLayers['fc2w'] = self.sess.run(self.weights['fc2'])
            self.retrieveLayers['fc2b'] = self.sess.run(self.biases['fc2'])
            
            print('Value saved\n')
            #self.save_model()            
            return True
        
        else:            
            return False
            
    def save_model(self,_path = "/tmp/model.ckpt"):
        saver = tf.train.Saver()
        self.save_path = saver.save(self.sess, _path)

    def load_model(self,component_name, component_shape, _path = "/tmp/model.ckpt"):
        
        retrieved_component = tf.get_variable(component_name ,shape = component_shape)                    
        #get_variable to be utilized before trying to attempt below portion#
        saver = tf.train.Saver()
        saver.restore(self.sess, "/tmp/model.ckpt")

        return retrieved_component
        
#HIDE this before release
'''
classifier = miniClassifier(meta.joinedData,meta.labelsOneHot)
classifier.train_init()
classifier.compile(100)
'''
#HIDE
