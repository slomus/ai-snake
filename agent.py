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
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # kolejka()
        self.model = Linear_QNet(11, 256, 3) #model sieci 11 wejsc z 256 neuronwami w ukrytej warstwie 3 wyjscia
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) #trener modelu

    #zbiera dane dotyczace aktualnego stanu gry na podstawie pozycji wezka, kierunku ruchu oraz lokalizacji jedzenia
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # sciana/cialo prosto ruch niebezpieczny
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # j.w. ale prawo
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # j.w. ale lewo
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # kierunek ruchu
            dir_l,
            dir_r,
            dir_u,
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
        #im wiecej gier zagalismy mamy coraz bardziej wytrenowany model wiec zmienjszamy 
        #nasza randomicjazje aby zacząć coraz bardziej wykorzystywać nasz model
        self.epsilon = 80 - self.nuber_games
        final_move = [0,0,0] #brak ruchu
        if random.randint(0, 200) < self.epsilon: #tutaj stawiamy na losowość
            move = random.randint(0, 2)
            final_move[move] = 1
        else: # wytrenowany model podejmuje decyzje o ruchu
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_0)
            move = torch.argmax(prediction).itme()
            final_move[move] = 1

        return final_move

#rozpoczecie gry i treningu modelu
def start_game_and_train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:

        #stare ruchy
        old_state = agent.get_state(game)

        #pobranie ruchy
        final_move = agent.get_action(old_state)

        reward, done, score = game.play_step(final_move)

        new_state = agent.get_state(game)

        agent.train_short_memory(old_state, final_move, reward, new_state, done)

        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            game.reset()
            agent.nuber_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()


            print("Gra", agent.nuber_games, "Punkty", score, "Rekord", record)
            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.nuber_games)
            plot(plot_scores, plot_mean_scores)
    

if __name__ == '__main__':
    start_game_and_train()