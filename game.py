import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN = (0, 255, 0)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 20

class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.last_position = deque(maxlen=5) #tylko 5 bo potem jak juz zje kilka jablek to sobie radzi zawsze na poczatku ma problemy zwiazane z krecieniem sie w "kółko"
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Wąż')
        self.clock = pygame.time.Clock()
        self.reset()

    #poczatkowa inicjalizacja stanu snaka
    #tym razem w funkcji zewnętrznej aby można było po przegranej grze przez model aktywować ja na nowo
    def reset(self, record=0):
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.last_position.clear()
        self.last_position.append(self.head)
        self.score = 0
        self.food = None
        self.food_positions = set()
        if record < 15:
            num_food_blocks = 5
        else:
            num_food_blocks = 1
        for _ in range(num_food_blocks):
            self._place_food()

        self.time_since_last_food = 0
        self.frame_iteration = 0


    #umieszczanie jedzeinia w dostepnych obszarza
    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            food = Point(x, y)
            if food not in self.snake and food not in self.food_positions:
                self.food_positions.add(food)
                break


    def play_step(self, action):
        self.frame_iteration += 1
        self.time_since_last_food += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        self.last_position.append(self.head)

        reward = 0
        game_over = False

        #sprawdzanie czy nie jest kolizja lub nie przekroczono 100 (mozna to zwiekszycz ale wtedy waz zyje dluzej i nauka trwa dluzej)
        #warunek moze tez wyglądac poprsotu tak if self.is_collision(): dzieki temu gra skoczy sie dopiero jak udezy sciane albo samego siebie
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        #sprawdzanie sekfencji ruchow weza aby nie krecil sie w "kółko"
        if all(pos == self.head for pos in self.last_position):
            reward -= 5
           #print("Wąż kręci się w kółko")

        if len(self.last_position) > 1 and self.last_position[0] == self.head and self.last_position[-1] == self.head:
            reward -= 5
            #print("Wąż kręci się w kółko")

        if self.head in self.food_positions:
            self.food_positions.remove(self.head)
            self.score += 1
            reward += 10
            if self.score < 15 and len(self.food_positions) < 50:
                self._place_food()
            else:
                self._place_food()
        else:
            self.snake.pop()

        if self.time_since_last_food >= 5 * SPEED:
            reward -= 10
            self.time_since_last_food = 0

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score



    def is_collision(self, pt=None):
        #print(f"PT: {pt}")
        if pt is None:
            pt = self.head
        return (pt.x >= self.w - BLOCK_SIZE or pt.x < 0 or #sprawdzanie sciany i ciala weza
            pt.y >= self.h - BLOCK_SIZE or pt.y < 0 or 
            pt in self.snake[1:])


    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        for food in self.food_positions:
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
        font = pygame.font.Font(None, 25)
        text = font.render("Wynik: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [prosto, prawo, lewo]
        # [1, 0, 0] -> prosto
        # [0, 1, 0] -> prawo
        # [0, 0, 1] -> lewo
        # p -> prawo
        # d -> dół
        # l -> lewo
        # g -> góra
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):  #idziemy prosto
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  #skręt w prawo
            new_dir = clock_wise[(idx + 1) % 4]
        else:  #skręt w lewo
            new_dir = clock_wise[(idx - 1) % 4]

        #zapobiegamy cofnięciu
        if (self.direction == Direction.RIGHT and new_dir == Direction.LEFT) or \
        (self.direction == Direction.LEFT and new_dir == Direction.RIGHT) or \
        (self.direction == Direction.UP and new_dir == Direction.DOWN) or \
        (self.direction == Direction.DOWN and new_dir == Direction.UP):
            new_dir = self.direction

        self.direction = new_dir
        x, y = self.head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)