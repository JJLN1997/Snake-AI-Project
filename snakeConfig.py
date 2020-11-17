
# size of a small grid, in term of pixle
gridSize = 30
# how many column & row in the grid 
gridWidth = 25
gridHeight = 25
# size of the screen
screenWidth = gridWidth*gridSize
screenHeight = gridHeight*gridSize
startPosition = [(12,12),(12,13),(12,14)]

maxStep = 200
growthRate = 1
up = (0,-1)
down = (0,1)
left = (-1,0)
right = (1,0)
Movements = [up,down,left,right]

# colour
colHead = (153, 204, 0)
colBody = (0, 179, 60)
colFood = (255, 77, 77)
colFoodBound = (204, 102, 153)
gridLight = (204, 204, 204)
gridDeep = (179, 179, 179)

allGrid = [(i,j) for i in range(0,25) for j in range(0,25)]

# AI parameters
# lv0 input to h1 6*9
# lv1 h1 to h2 6*7
# lv2 h2 to output 4*7
networkStruct = [("lv0",6,9),("lv1",6,7),("lv2",4,7)]
NNSize = 124

numBest = 5
populationSize = 100
Pmutate = 0.5
randomWalkDelta = 0.35