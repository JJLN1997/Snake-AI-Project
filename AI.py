from snakeConfig import *
import os
import numpy as np
import pandas as pd
import queue
import math
import random
### base line model
def greedyDecent(snake,food):
    Xs,Ys = snake.positions[0]
    Xf,Yf = food.position
    Q = queue.PriorityQueue()
    for i in range(0,len(Movements)):
        m = Movements[i]
        newX = Xs + m[0]     
        newY = Ys + m[1]
        if newX < 0 or newX > 24 or newY < 0 or newY > 24:
            continue
        elif (newX,newY) in snake.positions:
            continue
        
        heurstic = abs(newX-Xf) + abs(newY-Yf)
        Q.put((heurstic,i,m))

    if Q.empty(): return snake.direction
    else: return Q.get()[2]

def logBaseLineResult(snake):
    with open("baseLineResult.csv","a") as f:
        f.write(f"{snake.score}, {len(snake.positions)}\n")
### end baseline model

def CreateNextGeneration(perviousGen,cpuCore):

    def combineResults():
        combine = pd.read_csv(f"training/gen{perviousGen}_core0.csv")
        os.remove(f"training/gen{perviousGen}_core0.csv")
        # os.remove(f"training/gen{perviousGen}.csv")
        for c in range(1,cpuCore):
            path = f"training/gen{perviousGen}_core{c}.csv"
            temp = pd.read_csv(path)
            combine = combine.append(temp)
            os.remove(path)
        return combine
    
    def storeResult(previousGen,df):
        Scores = list(df["Score"])
        averageScore = round(sum(Scores)/len(Scores))
        df = df.nlargest(numBest,"Score")
        df.to_csv(f"training/gen{perviousGen}_res.csv",index=False)
        bestSnake = list(df.iloc[0])
        with open("NNResult.csv","a") as f:
            f.write(f"gen,{previousGen},{averageScore},{round(bestSnake[0])},{bestSnake[1:]}\n")
        return df

    def getBest(df):
        bs = pd.read_csv("training/bestSnake.csv")
        bs = bs.append(df)
        bs = bs.nlargest(numBest,"Score")
        bs.to_csv("training/bestSnake.csv",index=False)
        return bs

    df = combineResults()
    df = storeResult(perviousGen,df)

    col = df.columns

    midPop = round(populationSize/2)
    newPopulation = pd.DataFrame(columns = col)

    # uniform corssover
    for i in range(0,midPop):
        parents = df.sample(n=3,replace = False, weights = "Score")
        temp = [-1]
        for j in col[1:]:
            rInt = random.randrange(0,3)
            temp.append(parents.iloc[rInt][j])
        temp = pd.DataFrame([temp],columns=col)
        newPopulation = newPopulation.append(temp)
    
    # two point crossover
    for i in range(0,midPop):
        parents = df.sample(n=3,replace = False, weights = "Score")
        parents = parents.sample(frac = 1)

        splitPoints = random.sample(list(range(1,NNSize-1)),2)
        
        p1 = list(parents.iloc[0])[1:]
        p2 = list(parents.iloc[1])[1:]
        p3 = list(parents.iloc[2])[1:]

        temp = [-1] + p1[0:min(splitPoints)] + p2[min(splitPoints):max(splitPoints)] + p3[max(splitPoints):]
        temp = pd.DataFrame([temp],columns=col)
        newPopulation = newPopulation.append(temp)

    newPopulation = newPopulation.sample(frac=1)
    newPopulation.reset_index(drop=True,inplace=True)

    for i in range(0,populationSize):
        # whole mutation
        if i%2 == 0:
            for j in col[1:]:
                r = random.random()
                if r < Pmutate:
                    newPopulation.at[i,j] = random.random()*4-2
        # Random Walk
        else:
            for j in col[1:]:
                r = random.random()
                if r < Pmutate:
                    newPopulation.at[i,j] = random.sample([-1,1],1)[0]*randomWalkDelta + newPopulation.at[i,j]
    
    newPopulation = newPopulation.append(df) # add back the best snakes from last gen
    newPopulation = newPopulation.sample(frac=1)
    newPopulation.to_csv(f"training/gen{perviousGen+1}.csv",index=False)

class Agent():

    def __init__(self,weights):
        self.setup(weights)
        pass

    # return the result of the input layer
    def Sensor(self,snake,food):

        Xs,Ys = snake.positions[0]
        Xf,Yf = food.position
        
        delta = (Xf-Xs,Yf-Ys)
        normalize = max(abs(delta[0]),abs(delta[1]))
        if normalize != 0: delta = tuple([e/normalize for e in delta])
        else: delta = (0,0)
        # print(delta)

        directions = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
        result = [0 for i in range(0,8)]

        for i in range(0,len(directions)):
            d = directions[i]
            newX = Xs + d[0]     
            newY = Ys + d[1]

            if newX < 0 or newX > 24 or newY < 0 or newY > 24:
                result[i] = -1
            elif (newX,newY) in snake.positions:
                result[i] = -1
            elif d == delta: # align with food
                result[i] = 1

        return result
    
    def setup(self,weights):
        preIndex = 0
        self.Brain = {}
        self.weights = weights
        for tup in networkStruct:
            name,m,n = tup
            endIndex = preIndex + m*n
            self.Brain[name] = np.array(weights[preIndex:endIndex]).reshape(m,n)
            preIndex = endIndex

    def Predict(self,sensor):

        def Sigmoid(x,a = 1):
            return (2/(1+math.exp(-a*x)))-1

        f1 = np.vectorize(Sigmoid)
        res = np.array(sensor)

        for lev in range(0,3):
            res = np.append(res,1)
            res = np.matmul(self.Brain[f"lv{lev}"],res)
            res = f1(res)
        
        res = list(res)
        return Movements[res.index(max(res))]

    def logNNResult(self,score,gen,cpuCore):
        filePath = f"training/gen{gen}_core{cpuCore}.csv"

        # add header
        if not os.path.exists(filePath):
            with open(filePath,"w") as f:
                f.write("Score")
                for i in range(0,NNSize):
                    f.write(f",W{i}")
                f.write("\n")
        with open(filePath,"a") as f:
            f.write(f"{score}")
            for w in self.weights:
                f.write(f",{w}")
            f.write("\n")