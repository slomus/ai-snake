import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
from matplt import plot

MAX_MEMORY = 100_000 #Maksymalna liczba zapamiętanych doświadczeń gry !!! ZWIĘKSZENIE MOŻA SPOWODOWAC "ZAMULENIE" URZĄDZENIA !!!
BATCH_SIZE = 1000 #Liczba próbek/pokolen
LR = 0.001 #Współczynnik 


class Agent:
    
    def __init__(self):
        self.nuber_games = 0 #liczba gier
        self.epsilon = 0 # randomness
        self.gamma = 0.99 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # kolejka()
        self.model = Linear_QNet(11, 256, 3) #model sieci 11 wejsc z 256 neuronwami w ukrytej warstwie 3 wyjscia
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) #trener modelu

    #zbiera dane dotyczace aktualnego stanu gry na podstawie pozycji wezka, kierunku ruchu oraz lokalizacji jedzenia
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_p = Point(head.x + 20, head.y)
        point_g = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_p = game.direction == Direction.RIGHT
        dir_g = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # sciana/cialo prosto ruch niebezpieczny
            (dir_p and game.is_collision(point_p)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_g and game.is_collision(point_g)) or 
            (dir_d and game.is_collision(point_d)),

            # j.w. ale prawo
            (dir_g and game.is_collision(point_p)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_g)) or 
            (dir_p and game.is_collision(point_d)),

            # j.w. ale lewo
            (dir_d and game.is_collision(point_p)) or 
            (dir_g and game.is_collision(point_l)) or 
            (dir_p and game.is_collision(point_g)) or 
            (dir_l and game.is_collision(point_d)),
            
            # kierunek ruchu
            dir_l,
            dir_p,
            dir_g,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # jedzenie na lewo
            game.food.x > game.head.x,  # jedzenie na prawo
            game.food.y < game.head.y,  # jedzenie u góry
            game.food.y > game.head.y  # jedzenie na dole
            ]

        return np.array(state, dtype=int)


    #przechoduje nasze doświadczenie w pamieci które potem zostana wykorzystane przez nasz model, ograniczone przez liczbe mozliwych doswiadczen (MAX_MEMORY)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #jesli osiągniemy limit usuwamy od lewej czyli najstarszy tzw popleft() https://www.geeksforgeeks.org/deque-in-python/

    #dokonujemy szybkiego uczenia modelu na podstawie przeslanych danych
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    #uczy model na podstawie danych zapisanych w
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #pobieramy losowe doświadczenia z pamięci, jeśli liczba doświadczeń przekracza limit MAX_MEMORY.
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    

    #odpowiada za wybranie akcji (ruchu) 
    #dokładny opis działania ruchu plik game.py def _move
    def get_action(self, state):
        self.epsilon = max(0.01, self.epsilon * 0.995)  # Stopniowe zmniejszanie eksploracji
        final_move = [0, 0, 0]  # Brak ruchu

        if random.random() < self.epsilon:  # Losowy ruch
            move = random.randint(0, 2)
            final_move[move] = 1
        else:  # Ruch na podstawie modelu
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
