from PIL import Image
import numpy as np

filePath = 'D:/Documents/gitWork/NNStuff/Neural_Nets_Stuff/NST'
pathContent = filePath + '/Content.jpg'
pathStyle = filePath + '/Style.jpg'

imageContent = Image.open(pathContent)
imageStyle = Image.open(pathStyle)

#matching scales
imageStyle = imageStyle.resize(imageContent.size)
#numpy image

mainContent = np.array(imageContent,dtype = np.float32)
mainStyle = np.array(imageStyle,dtype =  np.float32)

transpose_order = [1,0]
if (mainContent.shape[2]>1):
    transpose_order += [2]

mainContent = mainContent.transpose(transpose_order)
mainStyle = mainStyle.transpose(transpose_order)

shapeImg = mainContent.shape

joinedData = np.stack([mainContent,mainStyle])
joinedData = (joinedData)/ 255

labels = np.array([x for x in range(joinedData.shape[0])])

labelsOneHot = np.identity(labels.shape[0])

