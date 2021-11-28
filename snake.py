# python libraries
from dataclasses import dataclass
import enum
from random import random
from enum import Enum
import threading
import time
import gym
import math
import numpy as np

# third party libraries
import turtle

LEFT = 'left'
RIGHT = 'right'
UP = 'up'
DOWN = 'down'

GAME_OVER_SCORE = -100
FOOD_SCORE = 25

EASY = 'easy'
MEDIUM = 'medium'
HARD = 'hard'

FOOD_COLOR = (51,255,51)
TAIL_COLOR = (163,163,163)
HEAD_COLOR = (0,0,0)

# there is no wall color, but we define it for the NN
WALL_COLOR = (255,51,221)

BACKGROUND_COLOR = (255,255,255)

OPPOSITES = {
    LEFT: RIGHT,
    RIGHT: LEFT,
    UP: DOWN,
    DOWN: UP
}

class StateAttributeType(Enum):
    CONVOLUTION = 0
    LINEAR = 1


@dataclass
class SnakeConfig():
    grid_size: int = 30
    grid_cell_size: int = 20
    difficulty: str = 'easy'
    is_human: bool = False
    debug: bool = False
    render: bool = True
    randomize_state: bool = True
    method: StateAttributeType = StateAttributeType.LINEAR

class SnakeEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    action_space = [0,1,2,3]

    direction_mapping = {
        0: UP,
        1: RIGHT,
        2: DOWN,
        3: LEFT
    }

    difficulties = {
        HARD: 0.05,
        MEDIUM: 0.1,
        EASY: 0.2
    }

    def __init__(self, conf: SnakeConfig):
        self.config = conf

        self.rng = np.random.default_rng(int(time.time()))
                
        self.move_lock = threading.Lock()
        self.debug = conf.debug

        if conf.difficulty in self.difficulties:
            self.sleep_time = self.difficulties[conf.difficulty]
        else:
            self.sleep_time = self.difficulties[EASY]

        # if agent is None we assume human mode
        self.is_human = conf.is_human
        self._render = conf.render
        self.randomize_state = conf.randomize_state

        self.GRID_SIZE = conf.grid_size
        self.GRID_CELL_WIDTH_PX = conf.grid_cell_size
        self.CELL_MAX = conf.grid_size*conf.grid_cell_size/2

        self.action_direction = {
            LEFT: -conf.grid_cell_size,
            RIGHT: conf.grid_cell_size,
            UP: conf.grid_cell_size,
            DOWN: -conf.grid_cell_size
        }
        
        self.tail=[]

        # turning is limited once a move has been made
        # this is to prevent doing a "quick" u-turn in between turns and
        # allowing the action of moving backwards aginst your tail
        self.can_turn = True
        # last direction used during "can_turn = False" is buffered and applied
        # to make the turning more smooth
        self.buffered_direction = None

        self.highest_score = 0
        self.score = 0

        ## TODO: Fix the game so if we are not in render mode that there's no grid stuff happening.
        ## At least do a proper reset with window.reset()
        self._initialize_window()
        self._initialize_snake()
        self._initialize_food()

        if self.is_human:
            self._create_window_bindings()

        # needs to be set after snake is initialized, allows dynamic state space if we add features
        state_space = self.get_state_features().flatten()
        self.state_space = len(state_space)

    def debug_print(self, message):
        if self.debug:
            print(message)

    def reset(self):
        self._game_over()

    def get_convolution_features(self):
        # add 2 to each grid size to represent walls
        features = np.ndarray(shape=(self.GRID_SIZE+2, self.GRID_SIZE+2, 3))
        min = int(-self.CELL_MAX - self.GRID_CELL_WIDTH_PX)
        max = int(self.CELL_MAX + 2*self.GRID_CELL_WIDTH_PX)
        for x in range(min, max, self.GRID_CELL_WIDTH_PX):
            for y in range(min, max, self.GRID_CELL_WIDTH_PX):
                (grid_x, grid_y) = self.get_grid_coord(x,y)
                if self.snake.xcor() == x and self.snake.ycor() == y:
                    features[grid_x, grid_y] = HEAD_COLOR
                elif self.food.xcor() == x and self.food.ycor() == y:
                    features[grid_x, grid_y] = FOOD_COLOR
                elif (self._wall_collision(x, y)):
                    print("wall collision")
                    features[grid_x, grid_y] = WALL_COLOR
                elif (self._tail_collision(x,y)):
                    features[grid_x, grid_y] = TAIL_COLOR
                else:
                    features[grid_x, grid_y] = BACKGROUND_COLOR

        return features

    def get_linear_features(self):
        x, y = self.snake.xcor(), self.snake.ycor()
        move_size = self.GRID_CELL_WIDTH_PX

        min_cell = -self.CELL_MAX+self.GRID_CELL_WIDTH_PX
        max_cell = self.CELL_MAX-self.GRID_CELL_WIDTH_PX

        wall_left = int(x - move_size < min_cell)
        wall_right = int(x + move_size > max_cell)
        wall_down = int(y - move_size < min_cell)
        wall_up = int(y + move_size > max_cell)

        body_left = int(self._tail_collision(x - move_size, y))
        body_right = int(self._tail_collision(x + move_size, y))
        body_down = int(self._tail_collision(x, y - move_size))
        body_up = int(self._tail_collision(x, y + move_size))

        food_left = int(self._acquired_food(x - move_size, y))
        food_right = int(self._acquired_food(x + move_size, y))
        food_down = int(self._acquired_food(x, y - move_size))
        food_up = int(self._acquired_food(x, y + move_size))

        food_to_the_left = int(x - self.food.xcor() >= 0)
        food_to_the_right = int(self.food.xcor() - x >= 0)
        food_downwards = int(y - self.food.ycor() >= 0)
        food_upwards = int(self.food.ycor() - y >= 0)

        # moving_left = int(self.snake.direction == LEFT)
        # moving_right = int(self.snake.direction == RIGHT)
        # moving_up = int(self.snake.direction == UP)
        # moving_down = int(self.snake.direction == DOWN)

        # x_dist = self.food.xcor() - x
        # y_dist = self.food.ycor() - y

        lst = [wall_left, wall_right, wall_down, wall_up, body_left, body_right, body_down, body_up, food_left, food_right, food_down, food_up, food_to_the_left, food_to_the_right, food_upwards, food_downwards]
        return np.asarray(lst)

    def get_state_features(self):
        if self.config.method == StateAttributeType.LINEAR:
            return self.get_linear_features()
        elif self.config.method == StateAttributeType.CONVOLUTION:
            return self.get_convolution_features()
        

    def step(self, action):
        # needs to return 
        self._turn(self.direction_mapping[action])
        reward = self.run_step()
        state_features = self.get_state_features()
        return state_features, reward, reward == GAME_OVER_SCORE

    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    def _get_random_x_y(self):
        x = int(self.rng.integers(-self.CELL_MAX, self.CELL_MAX) / self.GRID_CELL_WIDTH_PX) * self.GRID_CELL_WIDTH_PX
        y = int(self.rng.integers(-self.CELL_MAX, self.CELL_MAX) / self.GRID_CELL_WIDTH_PX) * self.GRID_CELL_WIDTH_PX
        return x, y

    def _place_food(self):
        x, y = self._get_random_x_y()

        while self._tail_collision(x, y) or x == self.snake.xcor() and y == self.snake.ycor():
            x, y = self._get_random_x_y()

        self.food.goto(x,y)

    def _generate_piece(self, shape, rgb_color1, rgb_color2=None ):
        piece = turtle.Turtle()

        piece.shape(shape)
        if rgb_color2:
            piece.color(rgb_color1, rgb_color2)
        else:
            piece.color(rgb_color1)

        piece.speed(0)

        piece.penup()
        return piece

    def _wall_collision(self, x, y):
        return (x > self.CELL_MAX 
                or y > self.CELL_MAX 
                or y < -self.CELL_MAX
                or x < -self.CELL_MAX)

    def _tail_collision(self, x, y):
        for tail_piece in self.tail:
            if x == tail_piece.xcor() and y == tail_piece.ycor():
                return True

        return False

    def _acquired_food(self, x, y):
        # acquire food
        distance = math.sqrt((x - self.food.xcor())**2 + (y - self.food.ycor())**2)
        return distance < self.GRID_CELL_WIDTH_PX
            
    def _create_new_tail_piece(self):
        new_tail_piece = self._generate_piece('square', TAIL_COLOR)

        self.tail.append(new_tail_piece)
        self.score += 1
        
        self._write_score()
        
        self.window.title(self.score)

    def _write_score(self, game_end=False):
        style = ('Courier', 13, 'bold')
        self.current_score_title.clear()
        self.current_score_title.write(f'Score: {self.score}', move=False, font=style, align='left')

        if game_end:
            self.top_score_title.clear()
            self.top_score_title.write(f'Top Score: {self.highest_score}', move=False, font=style, align='left')

    def _move_tail(self):
        for index in range(len(self.tail)-1,0,-1):
            x=self.tail[index-1].xcor()
            y=self.tail[index-1].ycor()
            self.tail[index].goto(x,y)

    def start_game(self):
        """ human entry point for the game """
        while True:
            self.run_step()
            time.sleep(self.sleep_time)

    def run_step(self):
        self._move_tail()

        prev_loc = (self.snake.xcor(), self.snake.ycor())
    
        #move the segment 0 to the head
        if len(self.tail) > 0:
            self.tail[0].goto(self.snake.xcor(),self.snake.ycor())

        self._move_snake()

        next_loc = (self.snake.xcor(), self.snake.ycor())

        if self._render:
            self.window.update()

        reward = None
        
        # unbuffer the direction that was stored for smoother turning in between sleep intervals
        if self.is_human and self.buffered_direction:
            self.move_lock.acquire()
            self._change_direction(self.buffered_direction)
            self.can_turn = False
            self.move_lock.release()
            self.buffered_direction = None

        # check for wall collision
        if self._wall_collision(self.snake.xcor(), self.snake.ycor()): 
            self._game_over()
            reward = GAME_OVER_SCORE

        # check for tail collision
        if self._tail_collision(self.snake.xcor(), self.snake.ycor()):
            self._game_over()
            reward = GAME_OVER_SCORE

        # acquire food
        if self._acquired_food(self.snake.xcor(), self.snake.ycor()):
            self._create_new_tail_piece()
            self._place_food()
            reward = FOOD_SCORE

        if not reward:
            food_loc = (self.food.xcor(), self.food.ycor())
            distance_before = math.sqrt((prev_loc[0]-food_loc[0])**2 + (prev_loc[1]-food_loc[1])**2)
            distance_after = math.sqrt((next_loc[0]-food_loc[0])**2 + (next_loc[1]-food_loc[1])**2)
            
            reward = 1 if distance_after < distance_before else -1

        self.debug_print(reward)
            
        return reward
            
    def _create_window_bindings(self):
        #binding
        self.window.listen()
        self.window.onkeypress(lambda: self._turn(UP),'w')
        self.window.onkeypress(lambda: self._turn(DOWN),'s')
        self.window.onkeypress(lambda: self._turn(RIGHT),'d')
        self.window.onkeypress(lambda: self._turn(LEFT),'a')

    def _initialize_window(self):
        # settings of the screen
        self.window=turtle.Screen()
        self.window.colormode(255)
        self.current_score_title = turtle.Turtle(visible=False)
        self.current_score_title.speed(0)
        self.current_score_title.goto(-self.CELL_MAX, self.CELL_MAX - 10)

        self.top_score_title = turtle.Turtle(visible=False)
        self.top_score_title.speed(0)
        self.top_score_title.goto(self.CELL_MAX - 125, self.CELL_MAX - 10)

        self._write_score(True)
        self.snake = None
        self.current_score_title.penup()
        
        self.window.title(str(self.score))
        self.window.bgcolor('white')
        width = (self.GRID_SIZE+1)*self.GRID_CELL_WIDTH_PX+(.75*self.GRID_CELL_WIDTH_PX)
        height = (self.GRID_SIZE+1)*self.GRID_CELL_WIDTH_PX+(.75*self.GRID_CELL_WIDTH_PX)
        self.window.setup(width=width,height=height)
        self.window.tracer(0)

    def _initialize_snake(self):
        #creating the head object
        if not self.snake:
            self.snake = self._generate_piece('square', HEAD_COLOR)

        if self.randomize_state:
            x, y = self._get_random_x_y()
        else:
            x, y = 0, 0

        self.snake.goto(x,y)

        self.snake.direction='freeze'

    def _initialize_food(self):
        #creating the food object
        self.food=turtle.Turtle()
        self.food.speed(0)
        self.food.shape('square')
        self.food.color(FOOD_COLOR)
        self.food.penup()

        self._place_food()

    def _change_direction(self, direction):
        if self.snake.direction != OPPOSITES[direction]:
            self.snake.direction = direction

    def _turn(self, direction):
        self.move_lock.acquire()

        if self.can_turn:
            self._change_direction(direction)

            if self.is_human:
                self.can_turn = False

        elif self.is_human and self.snake.direction != OPPOSITES[direction]:
            self.buffered_direction = direction

        self.move_lock.release()

    def _move_snake(self):
        x, y = self.get_cell_by_direction(self.snake.xcor(), self.snake.ycor(), self.snake.direction)
        self.snake.setx(x)
        self.snake.sety(y)
        self.can_turn = True

    def get_cell_by_direction(self, x, y, direction):
        if direction == UP or direction == DOWN:
            return (x, y + self.action_direction[direction])

        if direction == LEFT or direction == RIGHT:
            return (x + self.action_direction[direction], y)

    def _game_over(self):
        self._initialize_snake()

        # can't delete the pieces so we just hide them off grid
        for tail_piece in self.tail:
            tail_piece.reset()
            tail_piece.ht()
            # tail_piece.goto(self.CELL_MAX*2,self.CELL_MAX*2)

        self.tail.clear()

        if self.randomize_state:
            self.generate_random_tail()

        self._place_food()

        if self.score > self.highest_score:
            self.highest_score = self.score

        self.score = 0
        self._write_score(True)
        
        if self._render:
            self.window.update()

    def generate_random_tail(self):
        curx, cury = self.snake.xcor(), self.snake.ycor()

        for _ in range(self.rng.integers(0, 30)):
            possible_directions = [LEFT,RIGHT,DOWN,UP]
            direction = self.rng.choice(possible_directions, 1)[0]
            nextx, nexty = self.get_cell_by_direction(curx, cury, direction)

            while len(possible_directions) > 0 and (self._tail_collision(nextx, nexty) or self._wall_collision(nextx,nexty)):
                possible_directions.remove(direction)

                if len(possible_directions) == 0:
                    break

                direction = self.rng.choice(possible_directions, 1)[0]
                nextx, nexty = self.get_cell_by_direction(curx, cury, direction)

            # we generated our tail into a corner
            if len(possible_directions) == 0:
                break
            else:
                tail_piece = self._generate_piece('square', TAIL_COLOR)
                tail_piece.goto(nextx, nexty)
                curx = nextx
                cury = nexty
                self.tail.append(tail_piece)

    def get_pixel_coord(self,x, y):
        """ 
        get pixel x,y from grid coordinate
        """
        return (x * self.GRID_CELL_WIDTH_PX - self.CELL_MAX, y * self.GRID_CELL_WIDTH_PX - self.CELL_MAX)

    def get_grid_coord(self, x, y):
        """
        get grid x,y from pixel coordinate
        """
        return (int((x + self.CELL_MAX) / self.GRID_CELL_WIDTH_PX), int((y+ self.CELL_MAX) / self.GRID_CELL_WIDTH_PX))