import torch as tc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformer
import matplotlib.pyplot as plt
from tqdm import tqdm

class TransformerTrainer:

    def __init__(self, dataset: Dataset, embedding_dim, hidden_dim, n_heads, n_layers, dropout_p):
        obs_dim = dataset[0][0].shape[-1]
        self.dataset = dataset
        self.model = transformer.Transformer(obs_dim, embedding_dim, hidden_dim, 
                                             n_heads, n_layers, dropout_p)

    def train(self, n_epochs, lr, batch_size, batches_per_epoch):
        self.optimizer = tc.optim.Adam(self.model.parameters(), lr)
        self.loss_fn = nn.MSELoss()
        self.loss = []
        for epoch in tqdm(range(n_epochs)):
            dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
            for k, (obs, target) in enumerate(dataloader):
                self.optimizer.zero_grad()
                pred = self.model(obs)
                epoch_loss = self.loss_fn(pred, target)
                epoch_loss.backward()
                self.optimizer.step()
                self.loss.append(epoch_loss.detach().item())
                if k >= batches_per_epoch - 1:
                    break

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('transformer_loss.png')
        plt.close()

### Toy Datasets

class ReverseDataset(Dataset):
    def __init__(self, num_categories, seq_len, size):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size
        self.data = tc.randint(self.num_categories, size=(self.size, self.seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = tc.flip(inp_data, dims=(0,))
        return inp_data, labels
    
class SineDataset(Dataset):
    def __init__(self, dimensions, seq_len):
        super().__init__()
        self.dimensions = dimensions
        self.seq_len = seq_len
        time = tc.linspace(0, 100, 10000)
        coefficients = tc.linspace(0.1, 1.9, dimensions)
        domain = tc.outer(time, coefficients)
        self.data = tc.sin(domain)
        self.size = 10000 // seq_len - 1

    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        return self.data[i:i+self.seq_len], self.data[i+1:i+1+self.seq_len]
    
    def plot(self):
        plt.matshow(self.data.T, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.savefig('sine_dataset.png')
        plt.close()

if __name__=='__main__':

    dataset = SineDataset(5, 20)
    dataset.plot()
    trainer = TransformerTrainer(dataset, embedding_dim=10, hidden_dim=10, n_heads=1, n_layers=3, dropout_p=0.5)
    trainer.train(n_epochs=100, lr=1e-3, batch_size=10, batches_per_epoch=10)
    trainer.plot_loss()
    