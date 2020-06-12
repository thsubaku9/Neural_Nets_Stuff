import numpy as np

import meta
from minifier import *

def context_loss():
    return

def style_loss():
    return


GenImg = tf.Variable(initial_value = tf.random.uniform(shape = meta.shapeImg, minval = 0, maxval = 1,dtype = tf.float32), dtype = tf.float32, shape = meta.shapeImg, name = "GeneratedImg")

classifier = miniClassifier(meta.joinedData,meta.labelsOneHot)
classifier.train_init()
classifier.compile(50)

#create classifier
#create context loss
#create style loss
#assign weights
#create Generated image
#minimize total loss
#done
