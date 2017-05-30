import numpy as np
from fft import fftTransform

def preProcess(dataPoint, sampleRate, timePeriod):
    res = [[dataPoint[0][1],dataPoint[0][2],dataPoint[0][3]]]
    expLen = timePeriod / sampleRate + 1
    endTimeStamp = dataPoint[0][0] + timePeriod
    currentTimeStamp = dataPoint[0][0]
    endIndex = len(dataPoint)
    currentIndex = 0
    while currentTimeStamp < endTimeStamp:
        currentTimeStamp += sampleRate
        indexBefore = currentIndex
        while (currentIndex < endIndex and dataPoint[currentIndex][0] < currentTimeStamp):
            indexBefore = currentIndex # ensure that dataPoint[indexBefore][0] <= currentTimeStamp
            currentIndex += 1
        if currentIndex == endIndex:
            break # end of the array
        indexAfter = currentIndex #ensure that dataPoint[indexAfter][0] > currentTimeStamp
        pointBefore = dataPoint[indexBefore]
        pointAfter = dataPoint[indexAfter]
        timeGap = pointAfter[0] - pointBefore[0]
        timeStep = currentTimeStamp - pointBefore[0]
        x = pointBefore[1] + timeStep * (pointAfter[1] - pointBefore[1]) / timeGap
        y = pointBefore[2] + timeStep * (pointAfter[2] - pointBefore[2]) / timeGap
        z = pointBefore[3] + timeStep * (pointAfter[3] - pointBefore[3]) / timeGap
        res.append([x,y,z])
        currentIndex = indexBefore
    # fill in the points left
    resLen = len(res)
    lossPoint = expLen - resLen
    stepX = stepY = stepZ = 0
    for i in range(endIndex - lossPoint, endIndex):
        p1 = dataPoint[i]
        p2 = dataPoint[i - 1]
        stepX += (p1[1] - p2[1]) / (p1[0] - p2[0])
        stepY += (p1[2] - p2[2]) / (p1[0] - p2[0])
        stepZ += (p1[3] - p2[3]) / (p1[0] - p2[0])
    stepX = stepX / lossPoint * sampleRate
    stepY = stepY / lossPoint * sampleRate
    stepZ = stepZ / lossPoint * sampleRate
    while resLen < expLen:
        p = res[resLen - 1]
        res.append([p[0] + stepX, p[1] + stepY, p[2] + stepZ])
        resLen += 1
    return res

def timeFreqAnalysis(dataPoint):
    sampleRate = 10 # (ms)
    timePeriod = 4000 # 4s
    msPerSecond = 1000 # 1000ms = 1s
    res = preProcess(dataPoint, sampleRate, timePeriod)
    ans = fftTransform(res, msPerSecond / sampleRate, timePeriod / msPerSecond)
    return ans

if __name__ == '__main__':
    dataPoint = np.loadtxt(open("test.csv","r"),delimiter=",",skiprows=0)
    dataPoint = dataPoint.tolist()
    # res = preProcess(dataPoint, 10 ,4000)
    # fd = open('res.csv', 'w')
    # for point in res:
    #     line = ','.join(str(p) for p in point) + '\n'
    #     fd.write(line.encode('utf-8'))
    ans = timeFreqAnalysis(dataPoint)
    print ans
