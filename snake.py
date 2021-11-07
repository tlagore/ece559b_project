# python libraries
import threading
import time
import random

# third party libraries
import turtle

LEFT = 'left'
RIGHT = 'right'
UP = 'up'
DOWN = 'down'

EASY = 'easy'
MEDIUM = 'medium'
HARD = 'hard'

OPPOSITES = {
    LEFT: RIGHT,
    RIGHT: LEFT,
    UP: DOWN,
    DOWN: UP
}

class SnakeEnvironment():
    difficulties = {
        HARD: 0.05,
        MEDIUM: 0.1,
        EASY: 0.2
    }

    def __init__(self, grid_size, grid_cell_size, difficulty, agent=None):
        self.move_lock = threading.Lock()

        if difficulty in self.difficulties:
            self.sleep_time = self.difficulties[difficulty]
        else:
            self.sleep_time = self.difficulties[EASY]

        self.agent = agent

        self.GRID_SIZE = grid_size
        self.GRID_CELL_WIDTH_PX = grid_cell_size
        self.CELL_MAX = grid_size*grid_cell_size/2

        self.action_direction = {
            LEFT: -grid_cell_size,
            RIGHT: grid_cell_size,
            UP: grid_cell_size,
            DOWN: -grid_cell_size
        }
        
        self.tail=[]

        # turning is limited once a move has been made
        # this is to prevent doing a "quick" u-turn in between turns and
        # allowing the action of moving backwards aginst your tail
        self.can_turn = True
        # last direction used during "can_turn = False" is buffered and applied
        # to make the turning more smooth
        self.buffered_direction = None

        self.score = 0

        self._initialize_window()
        self._initialize_snake()
        self._initialize_food()

        if not self.agent:
            self._create_window_bindings()
    
    def _place_food(self):
        x = int(random.randint(-self.CELL_MAX, self.CELL_MAX) / self.GRID_CELL_WIDTH_PX) * self.GRID_CELL_WIDTH_PX
        y = int(random.randint(-self.CELL_MAX, self.CELL_MAX) / self.GRID_CELL_WIDTH_PX) * self.GRID_CELL_WIDTH_PX
        
        self.food.goto(x,y)

    def _generate_piece(self, colour, shape):
        piece = turtle.Turtle()

        piece.shape(shape)
        piece.color(colour)
        piece.speed(0)

        piece.penup()
        return piece

    def _wall_collision(self):
        return (self.snake.xcor() > self.CELL_MAX 
                or self.snake.ycor() > self.CELL_MAX 
                or self.snake.ycor() < -self.CELL_MAX
                or self.snake.xcor() < -self.CELL_MAX)

    def _tail_collision(self):
        for tail_piece in self.tail:
            if self.snake.xcor() == tail_piece.xcor() and self.snake.ycor() == tail_piece.ycor():
                return True

        return False

    def _acquired_food(self):
        # acquire food
        return self.snake.distance(self.food) < self.GRID_CELL_WIDTH_PX
            
    def _create_new_tail_piece(self):
        new_tail_piece = self._generate_piece('black', 'square')

        self.tail.append(new_tail_piece)
        self.score += 1
        self.window.title(self.score)

    def _move_tail(self):
        for index in range(len(self.tail)-1,0,-1):
            x=self.tail[index-1].xcor()
            y=self.tail[index-1].ycor()
            self.tail[index].goto(x,y)

    def start_game(self):
        while True:
            self.window.update()

            # check for wall collision
            if self._wall_collision(): 
                self._game_over()

            # check for tail collision
            if self._tail_collision():
                self._game_over()

            # acquire food
            if self._acquired_food():
                self._create_new_tail_piece()
                self._place_food()
                
            self._move_tail()

            #move the segment 0 to the head
            if len(self.tail) > 0:
                self.tail[0].goto(self.snake.xcor(),self.snake.ycor())

            self._move_snake()


            # unbuffer the direction that was stored for smoother turning in between sleep intervals
            if self.buffered_direction:
                self.snake.direction = self.buffered_direction
                self.buffered_direction = None

            time.sleep(self.sleep_time)

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
        self.window.title(str(self.score))
        self.window.bgcolor('white')
        width = (self.GRID_SIZE+1)*self.GRID_CELL_WIDTH_PX+(.75*self.GRID_CELL_WIDTH_PX)
        height = (self.GRID_SIZE+1)*self.GRID_CELL_WIDTH_PX+(.75*self.GRID_CELL_WIDTH_PX)
        self.window.setup(width=width,height=height)
        self.window.tracer(0)

    def _initialize_snake(self, starting_pos = None):
        #creating the head object
        self.snake = self._generate_piece('black', 'square')

        if starting_pos:
            self.snake.goto(starting_pos[0], starting_pos[1])
        else:
            self.snake.goto(0,0)

        self.snake.direction='freeze'

    def _initialize_food(self):
        #creating the food object
        self.food=turtle.Turtle()
        self.food.speed(0)
        self.food.shape('square')
        self.food.color('green')
        self.food.penup()

        self._place_food()

    def _turn(self, direction):
        self.move_lock.acquire()

        print(f"in turn with direction = {direction}, snake_direction={self.snake.direction}, can_turn={self.can_turn}")

        if self.snake.direction != OPPOSITES[direction] and self.can_turn:
            self.snake.direction = direction
            self.can_turn = False
        elif self.snake.direction != OPPOSITES[direction]:
            self.buffered_direction = direction
            print(f"can't move right now, buffered direction {direction}")

        print (f'set snake direction to {self.snake.direction}')

        self.move_lock.release()

    def _move_snake(self):
        if self.snake.direction == UP or self.snake.direction == DOWN:
            self.snake.sety(self.snake.ycor() + self.action_direction[self.snake.direction])
        
        if self.snake.direction==LEFT or self.snake.direction == RIGHT:
            self.snake.setx(self.snake.xcor() + self.action_direction[self.snake.direction])

        self.can_turn = True

    def _game_over(self):
        self.snake.goto(0,0)
        self.snake.direction='freeze'

        # can't delete the pieces so we just hide them off grid
        for tail_piece in self.tail:
            tail_piece.goto(self.CELL_MAX*2,self.CELL_MAX*2)

        self.tail.clear()
        self.score=0
        self.window.title(str(self.score))

    def get_pixel_coord(self,x, y):
        """ 
        get pixel x,y from grid coordinate
        """
        return (x * self.GRID_CELL_WIDTH_PX - 300, y * self.GRID_CELL_WIDTH_PX - 300)

    def get_grid_coord(self, head):
        """
        get grid x,y from pixel coordinate
        """
        return ((head.xcor() + 300) / self.GRID_CELL_WIDTH_PX, (head.ycor() + 300) / self.GRID_CELL_WIDTH_PX)

    def get_state_features(self):
        # up is wall
        # down is wall
        # right is wall
        # left is wall
        # up is food
        # down is food
        # 
        pass

class Agent():
    def __init__(self):
        """ """

    def get_action(self):
        pass

game = SnakeEnvironment(30, 20, MEDIUM)
game.start_game()

# wn.mainloop()
