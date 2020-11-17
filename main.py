import pygame
import sys
import random
import pandas as pd
from snakeConfig import *
from AI import *
# from utility import *
# pylint: skip-file
class Snake():
    def __init__(self):
        self.reset()
        if isTraining: self.agent = Agent(getSnake())
        else: self.agent = Agent([0 for i in range(0,124)])

    def getHeadPosition(self):
        return self.positions[0]

    def turn(self, point):
        if self.length > 1 and (point[0]*-1, point[1]*-1) == self.direction:
            # rejecting the reverse direction
            return
        else:
            self.direction = point

    def move(self):
        curX, curY = self.getHeadPosition()
        x,y = self.direction
        new = (curX+x,curY+y)

        if len(self.positions) > 2 and new in self.positions[2:]:
            # print("hit body")
            # logBaseLineResult(self)
            self.resetAgent()
            self.reset()
            return
        if new[0] < 0 or new[0] > 24 or new[1] < 0 or new[1] > 24:
            # print("hit boundary")
            # logBaseLineResult(self)
            self.resetAgent()
            self.reset()
            return
        if self.stepRemain < 0:
            # print("step limit reached")
            # logBaseLineResult(self)
            self.resetAgent()
            self.reset()
            return

        self.stepRemain -= 1
        self.step += 1
        self.positions.insert(0,new)
        if len(self.positions) > self.length:
            self.positions.pop()

        # eating food
        if self.getHeadPosition() == self.refFood.position:
            self.length += growthRate
            self.eaten += growthRate
            self.maxStep += growthRate
            self.stepRemain = self.maxStep
            self.refFood.randomizePosition()

    def reset(self):
        self.positions = startPosition.copy()
        self.length = len(startPosition)
        self.direction = up
        self.step = 0
        self.eaten = 0
        self.score = 0
        self.maxStep = maxStep
        self.stepRemain = self.maxStep

    def resetAgent(self):
        if isTraining:
            self.agent.logNNResult(self.score,trainingGen,cpuCore)
            Weights = getSnake()
            if Weights != None: # when weight is none the training set is finished
                self.agent.setup(Weights)

    def calScore(self): # the fitness fn
        self.score = self.step + self.eaten**3

    def draw(self,surface):
        for i in range(len(self.positions)-1,-1,-1):
            r = pygame.Rect((self.positions[i][0]*gridSize, self.positions[i][1]*gridSize), (gridSize,gridSize))
            if (i == 0): pygame.draw.rect(surface, colHead, r)
            else: pygame.draw.rect(surface, colBody, r)
            pygame.draw.rect(surface, gridLight, r, 1)

    def handleKeys(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.turn(up)
                elif event.key == pygame.K_s:
                    self.turn(down)
                elif event.key == pygame.K_a:
                    self.turn(left)
                elif event.key == pygame.K_d:
                    self.turn(right)

    def selfControl(self):
        for event in pygame.event.get():
            pass
        # d = greedyDecent(self,self.refFood)

        s = self.agent.Sensor(self,self.refFood)
        d = self.agent.Predict(s)
        self.turn(d)

class Food():
    def __init__(self,snake):
        self.position = (0,0)
        self.refSnake = snake # a ref to snake
        snake.refFood = self # give snake a ref to food
        self.randomizePosition()


    def randomizePosition(self):
        avaPos = allGrid.copy()
        for i in self.refSnake.positions:
            avaPos.remove(i)
        self.position = random.sample(avaPos,1)[0]

    def draw(self, surface):
        x,y = self.position
        r = pygame.Rect((x*gridSize, y*gridSize), (gridSize, gridSize))
        pygame.draw.rect(surface, colFood, r)
        pygame.draw.rect(surface, colFoodBound, r, 3)


def drawGrid(surface):
    for y in range(0, gridHeight):
        for x in range(0, gridWidth):
            if (x+y)%2 == 0:
                r = pygame.Rect((x*gridSize, y*gridSize), (gridSize,gridSize))
                pygame.draw.rect(surface,gridDeep, r)
            else:
                rr = pygame.Rect((x*gridSize, y*gridSize), (gridSize,gridSize))
                pygame.draw.rect(surface,gridLight, rr)

# return a list of parameters represent a snake
def getSnake():
    global curSnake,trainingGen,population,isGameOver
    if isKeyBoard: return None
    else:
        if curSnake >= endIndex:
            isGameOver = True
            # print(f"training of {trainingGen} is finished")
            # curSnake = 0
            # CreateNextGeneration(trainingGen)
            # trainingGen += 1
            # population = pd.read_csv(f"training/gen{trainingGen}.csv")
        else:
            p = list(population.iloc[curSnake])[1:]
            print(curSnake)
            curSnake +=1
            return p

### global variables ###
isGameOver = False
isKeyBoard = False
isTraining = False

trainingGen = None
curSnake = None
endIndex = None
population = None
cpuCore = None
### global variables end ###

def init():
    global isKeyBoard,isTraining,trainingGen,curSnake,endIndex,population,cpuCore
    note = "require one of:\n -training gen# startIndex endIndex cpuCore \n -keyboard"
    arguments = sys.argv
    if len(arguments) < 2:
        print(f"Not enough arg\n{note}")
    elif arguments[1] == "-keyboard":
        isKeyBoard = True
        main()
    elif arguments [1] == "-training":
        isTraining = True
        trainingGen = int(arguments[2])
        curSnake = int(arguments[3])
        endIndex = int(arguments[4])
        cpuCore = int(arguments[5])
        population = pd.read_csv(f"training/gen{trainingGen}.csv")
        main()
    else:
        print(f"Wrong arg\n{note}")

def main():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((screenWidth, screenHeight), 0, 32)
    # pygame.display.iconify()
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    drawGrid(surface)
    snake = Snake()
    food = Food(snake)
    myfont = pygame.font.SysFont("monospace",16)
    while (not isGameOver):
        if isKeyBoard: 
            clock.tick(10)
            snake.handleKeys()
        elif isTraining:
            clock.tick(-1)
            snake.selfControl()
        snake.move()
        snake.calScore()
        drawGrid(surface)
        food.draw(surface)
        snake.draw(surface)
        screen.blit(surface, (0,0))
        textScore = myfont.render(f"Step:{snake.step} Apple:{snake.eaten} Score:{snake.score} Limit:{snake.stepRemain}",1, (0,0,0))
        textPos = myfont.render(f"Head:{snake.positions[0]}",1, (0,0,0))
        screen.blit(textScore, (10,10))
        screen.blit(textPos, (10,30))
        pygame.display.update()

if __name__ == "__main__":
    init()
