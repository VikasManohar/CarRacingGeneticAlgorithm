import pygame
import random
import os
import time
import neat
# import visualize
import pickle
import numpy as np
DRAW_LINES = True
pygame.font.init()
MAX_SCORE = 0
display_width = 600
display_height = 600
# thingc = None
black = (0,0,0)
white = (255,255,255)
red = (255,0,0)

car_width = 50
car_height = 75

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption("pyGame Test")
gameDisplay.fill(black)


gen = 0

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move_left(self):
        self.x += -5
    
    def move_right(self):
        self.x += 5
        
    def dont_move(self):
        pass

    def draw(self, gameDisplay):
        pygame.draw.rect(gameDisplay, red, [self.x,self.y,car_width,car_height])

class things:
    def __init__(self, thingx, thingy, thingw, thingh, thing_speed):
        self.thingx = thingx
        self.thingy = thingy
        self.thingw = thingw
        self.thingh = thingh
        self.thing_speed = thing_speed

    def move_thing(self):
        self.thingy += self.thing_speed
    
    def draw(self, gameDisplay):
        pygame.draw.rect(gameDisplay, white, [self.thingx, self.thingy, self.thingw, self.thingh])
        

def draw_score(gameDisplay, score):
    font = pygame.font.SysFont(None, 25)
    text = font.render("Dodged: " + str(score), True, white)
    gameDisplay.blit(text, (0,0))

def draw_generations(gameDisplay, gen):
    font = pygame.font.SysFont(None, 25)
    text = font.render("Generation: " + str(gen), True, white)
    gameDisplay.blit(text, (0,30))

def draw_cars_alive(gameDisplay,cars):
    font = pygame.font.SysFont(None, 25)
    text = font.render("Cars Alive: " + str(cars), True, white)
    gameDisplay.blit(text, (0,60))

def draw_max_score(gameDisplay):
    font = pygame.font.SysFont(None, 25)
    text = font.render("Max Dodged so far: " + str(MAX_SCORE), True, white)
    gameDisplay.blit(text, (0,90))

def draw_car_lines(gameDisplay, cars, thing):
    for car in cars:
            # draw lines from car to thing
            if DRAW_LINES:
                try:
                    pygame.draw.line(gameDisplay, (255,0,0), (car.x, car.y), (thing.thingx, thing.thingy + thing.thingh) , 1)
                    pygame.draw.line(gameDisplay, (0,255,0), (car.x, car.y), (thing.thingx + thing.thingw, thing.thingy + thing.thingh) , 1)
                    pygame.draw.line(gameDisplay, (0,0,255), (car.x + car_width, car.y), (thing.thingx, thing.thingy + thing.thingh), 1)
                    pygame.draw.line(gameDisplay, (255,255,255), (car.x + car_width, car.y), (thing.thingx + thing.thingw, thing.thingy + thing.thingh), 1)
                    
                except Exception as e:
                    print(e)
                    
            car.draw(gameDisplay)

def draw_window(gameDisplay, cars, thing, score, gen):
    gameDisplay.fill(black)

    for car in cars:
        car.draw(gameDisplay)

    thing.draw(gameDisplay)
    draw_car_lines(gameDisplay, cars, thing)
    draw_cars_alive(gameDisplay, str(len(cars)))
    draw_score(gameDisplay, score)
    draw_max_score(gameDisplay)
    draw_generations(gameDisplay, gen)
    pygame.display.update()

def softmax(x):
    print("***********input*************")
    print(x)
    print("***********input*************")
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def intersects(self, other):
    return not (self.top_right.x < other.bottom_left.x or self.bottom_left.x > other.top_right.x or self.top_right.y < other.bottom_left.y or self.bottom_left.y > other.top_right.y)

def eval_genomes(genomes, config):
    nets = []
    cars = []
    ge = []

    global gen, gameDisplay, MAX_SCORE
    gen += 1
    # if gen ==2: 
    #     quit()
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car((display_width * 0.45),(display_height * 0.85)))
        ge.append(genome)
    
    thingc = things(random.randrange(0, display_width -100), -600, 100, 100, 20)
    clock = pygame.time.Clock()
    
    score = 0

    run = True

    while run and len(cars) > 0 :
        clock.tick(500)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
        
        for x, car in enumerate(cars):  # give each car a fitness of 0.1 for each frame it stays alive
            ge[x].fitness += 0.1
            # print(ge[x].fitness)
            # (car.x  - (thingc.thingx + thingc.thingw))
            # ((car.x + car_width) - thingc.thingx)
            # (car.x - thingc.thingx)
            # ((car.x + car_width) - (thingc.thingx + thingc.thingw))
            # inputs = ((car.x  - (thingc.thingx + thingc.thingw)), 
            # ((car.x + car_width) - thingc.thingx), (car.x - thingc.thingx), ((car.x + car_width) - (thingc.thingx + thingc.thingw)), (car.y + car_height), (thingc.thingy + thingc.thingh))
            inputs = ( ((car.x + car_width) - thingc.thingx ), (car.x - (thingc.thingx + thingc.thingw)), ((car.y + car_height) - thingc.thingy), (car.y - (thingc.thingy + thingc.thingh)))
            output = nets[cars.index(car)].activate(inputs)
            # print(inputs)
            # output.m
            # print(output)
            # if output == 0:
            #     car.move_left()
            # elif output == 1:
            #     car.dont_move()
            # else:
            #     car.move_right()
            if output[0] > 0.5:
                car.move_right()
            elif output[0] < -0.5:
                car.move_left()
            else:
                car.dont_move()
            # else:
            #     car.dont_move()

        thingc.move_thing()

        #car hits the thing
        for car in cars:
            if car.y < thingc.thingy + thingc.thingh:
                if car.x > thingc.thingx and car.x < thingc.thingx + thingc.thingw or car.x + car_width > thingc.thingx and car.x + car_width < thingc.thingx + thingc.thingw:
                    ge[cars.index(car)].fitness -= 1
                    nets.pop(cars.index(car))
                    ge.pop(cars.index(car))
                    cars.pop(cars.index(car))

            
        #car goes out of play area
        for car in cars:
            if car.x > display_width - car_width or car.x < 0:
                ge[cars.index(car)].fitness -= 1
                nets.pop(cars.index(car))
                ge.pop(cars.index(car))
                cars.pop(cars.index(car))

        #car has dodged the thing
        if thingc.thingy > display_height:
            # print("inside check")
            thingc.thingy = 0 - thingc.thingh
            thingc.thingx = random.randrange(0, display_width)
            score += 1
            if score > MAX_SCORE:
                MAX_SCORE = score
            # can add this line to give more reward for dodging a thing
            for genome in ge:
                genome.fitness += 5
        
        

        draw_window(gameDisplay, cars, thingc, score, gen)


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # config.genome_config.add_activation('softmax', softmax)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
     #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 1000)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))
    print("++++++++++++++MAX_SCORE++++++++++++++++++")
    print(MAX_SCORE)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
