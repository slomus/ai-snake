import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os



class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.2) #zabezpieczenie przed overfittingu
        self.linear2 = nn.Linear(hidden_size, output_size)

    #przepływ danych w sieci
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    #zapis danych
    def save(self, file_name='model_data.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

#Odpowiada za trening sieci neuronowej
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    #tenuje model za pomoca torch.tensor (1 raz)
    def train_step(self, state, action, reward, next_state, done):
        #konwersja wejściowych danych na tensory
        state = torch.tensor(np.array(state), dtype=torch.float)  #stan bieżący
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)  #następny stan
        action = torch.tensor(np.array(action), dtype=torch.float)  #akcja w stanie
        reward = torch.tensor(np.array(reward), dtype=torch.float)  #nagroda

        #przekształcenie danych wejściowych do wymiaru (1, x)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)  #upewnij się, że done jest krotką

        #przewidywanie wartości Q
        pred = self.model(state)

        #kopia Q i obliczanie docelowej wartości
        target = pred.clone()

        #konwersja done do tensora
        done_tensor = torch.tensor(done, dtype=torch.bool)

        #upewnij się, że action jest w formacie indeksów
        action_indices = torch.argmax(action, dim=1)

        #obliczanie nowych wartości Q
        Q_new = reward + (self.gamma * torch.max(self.model(next_state), dim=1)[0] * ~done_tensor)

        #aktualizacja Q w docelowym tensorze (target[action_indices])
        target[range(len(done)), action_indices] = Q_new

        #optymalizacja straty
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        #optymalizacja wag
        self.optimizer.step()


