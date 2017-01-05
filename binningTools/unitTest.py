from binningExt import DownSample
import numpy as np

inputArray = np.ones((1000, 1000))
outputWidth = 10
outputHeight = 10
factor = (1000*1000)/(10.*10)

# 11 instead of 10, as we need the end points
yInd, xInd = np.indices((11, 11), dtype=float)

verticies = xInd.flatten().tolist() + yInd.flatten().tolist()

DSFunctor = DownSample(inputArray, outputWidth, outputHeight)

small = DSFunctor(verticies)

truth = np.ones((outputHeight, outputWidth))*factor

assert((small == truth).all())

newYInd = np.zeros((20, 20))
yvec = np.repeat(yInd[:, 0], 2)[1:-1]
for i in range(20):
    newYInd[:, i] = yvec

newXInd = np.zeros((20, 20))
xvec = np.repeat(xInd[0, :], 2)[1:-1]
for i in range(20):
    newXInd[i, :] = xvec

newXInd[:, 1] = 0.9

newVerticies = newXInd.flatten().tolist() + newYInd.flatten().tolist()

newDSFunctor = DownSample(inputArray, outputWidth, outputHeight, disjoint=True)
newSmall = newDSFunctor(newVerticies)
