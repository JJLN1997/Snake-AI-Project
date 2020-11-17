import math
import subprocess
from snakeConfig import populationSize,numBest
from AI import CreateNextGeneration

trainingGen = 0
endGen = 200
cpuCore = 4
totalSnake = populationSize + numBest
splitSize = math.ceil(totalSnake/cpuCore)

while(trainingGen <= endGen):
    processList = []
    for core in range(0,cpuCore):
        start = core*splitSize
        end = start + splitSize
        # print(start,end)
        if end > totalSnake: end = totalSnake 
        p = subprocess.Popen(f"python main.py -training {trainingGen} {start} {end} {core}",
                            stdout = subprocess.DEVNULL)
        processList.append(p)

    for p in processList:
        p.wait()

    print(f"Training in gen {trainingGen} is done")
    CreateNextGeneration(trainingGen,cpuCore)
    trainingGen += 1