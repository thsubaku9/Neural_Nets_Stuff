from PIL import Image
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

filePath = 'D:/Documents/gitWork/NNStuff/Neural_Nets_Stuff/NST'
pathContent = filePath + '/Content.jpg'
pathStyle = filePath + '/Style.jpg'

imageContent = Image.open(pathContent)
imageStyle = Image.open(pathStyle)

#matching scales
imageContent = imageContent.resize(imageStyle.size)
#imageStyle = imageStyle.resize(imageContent.size)
#numpy image

mainContent = np.array(imageContent,dtype = np.float32) / 255
mainStyle = np.array(imageStyle,dtype =  np.float32) / 255

'''
transpose_order = [1,0]
if (mainContent.shape[2]>1):
    transpose_order += [2]

mainContent = mainContent.transpose(transpose_order)
mainStyle = mainStyle.transpose(transpose_order)
'''

shapeImg = mainContent.shape
arr_noisy = [mainStyle,mainContent,]; bench_val = 6000
g_1 = tf.Graph()
g_1.as_default();g_1.device('/cpu:0');

tmp = np.reshape(mainContent,(1,mainContent.shape[0],mainContent.shape[1],mainContent.shape[2]))

G = tf.Variable(initial_value = tf.random.uniform(shape = (1,shapeImg[0],shapeImg[1],shapeImg[2]), minval = 0, maxval = 1,dtype = tf.float32), dtype = tf.float32, shape = (1,shapeImg[0],shapeImg[1],shapeImg[2]), name = "NoisyImg")
loss = tf.reduce_sum(tf.square(tf.subtract(G,tmp[0])))
opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#GPU fix
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
#GPU fix

with tf.Session(config = config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(0,100):    
        ls = sess.run(loss)
        sess.run(opt)
        clipper = tf.clip_by_value(G,0.0,1.0)
        G.assign(clipper)
        if(i%5 == 0):
            print(ls)
            img = sess.run(clipper)
            #plt.imshow(img[0])
            #plt.show()
            if(ls<= bench_val):
                arr_noisy.append(img[0])
####

joinedData = np.stack(arr_noisy)
labels = np.array([x for x in range(joinedData.shape[0])])

labelsOneHot = np.zeros((len(labels),5))
labelsOneHot[0][0] = 1
for x in range(1,len(arr_noisy)):
    labelsOneHot[x][1] = 1

