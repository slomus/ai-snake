from game import SnakeGame
from agent import Agent
from matplt import plot

def train():
    # Inicjalizacja gry i agenta
    agent = Agent()
    game = SnakeGame()
    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    while True:
        #pobieranie aktualneog stanu gry
        state_old = agent.get_state(game)

        #decyzja agenta
        action = agent.get_action(state_old)

        #wykonanie kroku w grze
        reward, game_over, score = game.play_step(action)

        #pobieranie nowego stanu
        state_new = agent.get_state(game)

        #zapamietywanie i trenownie szybkiej pamieci modelu
        agent.remember(state_old, action, reward, state_new, game_over)
        agent.train_short_memory(state_old, action, reward, state_new, game_over)

        #jesli koniec gry odpalamy na nowo
        if game_over:
            game.reset(record)
            agent.nuber_games += 1
            agent.train_long_memory()

            #aktualizacja randomizzacji 
            agent.epsilon = max(0.01, agent.epsilon * 0.995)
            #zapisywanie wyniku
            if score > record:
                record = score
                agent.model.save()

            print(f"Gra {agent.nuber_games}, Wynik: {score}, Rekord: {record}")

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.nuber_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)

if __name__ == "__main__":
    train()
