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

mainContext = np.array(imageContext)
mainStyle = np.array(imageStyle)
