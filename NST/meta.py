from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

filePath = 'D:/Documents/gitWork/NNStuff/Neural_Nets_Stuff/NST'
pathContent = filePath + '/Content.jpg'
pathStyle = filePath + '/Style.jpg'

imageContent = Image.open(pathContent)
imageStyle = Image.open(pathStyle)

#matching scales
imageContent = imageContent.resize((256,256))
imageStyle = imageStyle.resize((256,256))
#numpy image

mainContent = np.array(imageContent,dtype = np.float32) #/ 255
mainStyle = np.array(imageStyle,dtype =  np.float32) #/ 255

'''
transpose_order = [1,0]
if (mainContent.shape[2]>1):
    transpose_order += [2]

mainContent = mainContent.transpose(transpose_order)
mainStyle = mainStyle.transpose(transpose_order)
'''

shapeImg = mainContent.shape
arr_noisy = [mainStyle,mainContent,]; bench_val = 4000; bench_val2 = 4000
labels = [0,1]
g_1 = tf.Graph()
g_1.as_default();g_1.device('/cpu:0');

tmp = np.reshape(mainContent,(1,mainContent.shape[0],mainContent.shape[1],mainContent.shape[2]))
tmp2 = np.reshape(mainStyle,(1,mainStyle.shape[0],mainStyle.shape[1],mainStyle.shape[2]))


G = tf.Variable(initial_value = tf.random.uniform(shape = (1,shapeImg[0],shapeImg[1],shapeImg[2]), minval = 0, maxval = 255,dtype = tf.float32), dtype = tf.float32, shape = (1,shapeImg[0],shapeImg[1],shapeImg[2]), name = "NoisyImg")
loss = tf.reduce_sum(tf.square(tf.subtract(G,tmp[0])))
opt = tf.train.GradientDescentOptimizer(0.08).minimize(loss)

G2 = tf.Variable(initial_value = tf.random.uniform(shape = (1,shapeImg[0],shapeImg[1],shapeImg[2]), minval = 0, maxval = 255,dtype = tf.float32), dtype = tf.float32, shape = (1,shapeImg[0],shapeImg[1],shapeImg[2]), name = "NoisyImg_Style")
loss2 = tf.reduce_sum(tf.square(tf.subtract(G2,tmp2[0]))) 
opt2 = tf.train.GradientDescentOptimizer(0.08).minimize(loss2)

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
        clipper = tf.clip_by_value(G,0.0,255.0)
        G.assign(clipper)
        if(i%5 == 0):
            print(ls)
            img = sess.run(clipper)
            #plt.imshow(img[0])
            #plt.show()
            if(ls<= bench_val):
                arr_noisy.append(img[0])
                labels.append(1)
            elif(ls <= 2 *bench_val and ls >= 1.1 * bench_val):
                arr_noisy.append(img[0])
                labels.append(2)
        #if(i%20 == 0):
            #plt.imshow(img[0])
            #plt.show()            
    
        ls2 = sess.run(loss2)
        sess.run(opt2)
        clipper2 = tf.clip_by_value(G2,0.0,255.0)
        G2.assign(clipper2)
        if(i%5 == 0):
            print(ls2)
            img2 = sess.run(clipper2)
            #plt.imshow(img[0])
            #plt.show()
            if(ls2 <= bench_val2):
                arr_noisy.append(img2[0])
                labels.append(0)
            elif(ls <= 2 *bench_val2 and ls >= 1.1 * bench_val2):
                arr_noisy.append(img2[0])
                labels.append(2)
        #if(i%20 == 0):
            #plt.imshow(img2[0])
            #plt.show()

joinedData = np.stack(arr_noisy)

labelsOneHot = np.zeros((len(labels),3))
for x in range(0,len(labels)):
    labelsOneHot[x][labels[x]] = 1

