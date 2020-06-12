from PIL import Image
import numpy as np

filePath = 'D:/Documents/gitWork/NNStuff/Neural_Nets_Stuff/NST'
pathContext = filePath + '/Context.jpg'
pathStyle = filePath + '/Style.jpg'

imageContext = Image.open(pathContext)
imageStyle = Image.open(pathStyle)

#matching scales
imageStyle = imageStyle.resize(imageContext.size)
#numpy image

mainContext = np.array(imageContext,dtype = np.float32)
mainStyle = np.array(imageStyle,dtype =  np.float32)

transpose_order = [1,0]
if (mainContext.shape[2]>1):
    transpose_order += [2]

mainContext = mainContext.transpose(transpose_order)
mainStyle = mainStyle.transpose(transpose_order)

shapeImg = mainContext.shape

joinedData = np.stack([mainContext,mainStyle])
joinedData = (joinedData)/ 255

labels = np.array([x for x in range(joinedData.shape[0])])

labelsOneHot = np.identity(labels.shape[0])

