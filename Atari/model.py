import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, hidden_conv_layer_dims, hidden_lin_layer_dims) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for in_channels, out_channels, kernel_size, stride, padding in hidden_conv_layer_dims:
            self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dtype=torch.float64))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Flatten())
        for in_size, out_size in hidden_lin_layer_dims:
            self.layers.append(nn.Linear(in_features=in_size, out_features=out_size, dtype=torch.float64))
            self.layers.append(nn.ReLU())
        self.layers = self.layers[:-1]
        self.layers.append(nn.Softmax())
        self.optimizer = optim.Adam(self.parameters(), lr = 0.001)
    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        x = x.to(torch.float64)
        for layer in self.layers:
            # print(x.shape)
            x = layer(x)
        return x
    def _train__instance__(self, train_dataset) -> None:
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
        num_epochs = 5
        self.train()
        for epoch in tqdm(range(num_epochs)):
            for batch_id, (data, returns) in enumerate(train_loader):
                # print("here")
                states, actions = data[:][0], data[:][1]
                self.optimizer.zero_grad()
                # print(data[0])
                # print(states.shape, actions.shape)
                output = self(states)
                output_actions = output.gather(1, actions.unsqueeze(1))
                # print(output.shape, actions.shape, output_actions.shape)
                # raise("Pause")
                loss = torch.sum(torch.log(output_actions)*returns)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    def _save__model__(self, msg : str):
        torch.save(self, f"epoch_{msg}_model.pth")
    def _return__layers__(self):
        return self.layers
    

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        # if isinstance(data[0], torch.Tensor):
        #     data[0] = data[0].to(torch.float64)
        # if isinstance(data[1], torch.Tensor):
        #     data[1] = data[1].to(torch.float64)
        if isinstance(labels, torch.Tensor):
            labels = labels.to(torch.float64)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print("Requested : ", index, " | Available - data : ", len(self.data), ", labels : ", len(self.labels))
        sample = self.data[index]
        label = self.labels[index]
        return sample, label
