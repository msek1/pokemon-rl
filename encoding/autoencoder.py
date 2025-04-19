import torch
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot

# from process_csv import moves_df, pokemon_df
if __name__ == "__main__":
    from create_encoding_datasets import create_pokemon_dataset_tensor, create_move_dataset_tensor
else:
    from encoding.create_encoding_datasets import create_pokemon_dataset_tensor, create_move_dataset_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

EMBEDDING_DIMENSION = 10

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.4):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCELoss()  # For handling 0-heavy sparse data
        self.mse = nn.MSELoss()  # For reconstructing decimal values
        self.alpha = alpha

    def forward(self, outputs, targets):
        return self.alpha * self.bce(outputs, targets) + (1 - self.alpha) * self.mse(outputs, targets)

class Autoencoder(nn.Module):
    def __init__(self, input_dim = 26, latent_dim = 10):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.05),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.05),
            nn.Linear(32, latent_dim),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.05),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.05),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(model, data, num_epochs = 200, batch_size = 20):
    # loss_fn = HybridLoss()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_dataset = TensorDataset(data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for batch in train_loader:
            x_batch = batch[0]
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs[1], x_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
    
    pyplot.plot(range(1, num_epochs + 1), losses)
    pyplot.show()
    return losses

def get_latent_moves():
    # moves_tensor = torch.tensor(moves_df.values.astype(float), dtype=torch.float32)
    # moves_tensor = torch.nan_to_num(moves_tensor, nan=0.0)
    moves_tensor = create_move_dataset_tensor().to(DEVICE)

    autoencoder_moves = Autoencoder(input_dim = 27).to(DEVICE)
    losses_moves = train_autoencoder(autoencoder_moves, moves_tensor,1000,50)

    autoencoder_moves.eval()
    with torch.no_grad():
        latent_vectors, output_vectors = autoencoder_moves(moves_tensor)

    torch.save(autoencoder_moves, "autoencoder_moves.pth")
    print(f"Final Loss: {losses_moves[-1]}")

    return latent_vectors.cpu(), output_vectors.cpu()

def get_latent_pokemon():
    # pokemon_tensor = torch.tensor(pokemon_df.values.astype(float), dtype=torch.float32)
    # pokemon_tensor = torch.nan_to_num(pokemon_tensor, nan=0.0)
    pokemon_tensor = create_pokemon_dataset_tensor().to(DEVICE)

    autoencoder_pokemon = Autoencoder(input_dim = 44).to(DEVICE)
    losses_pokemon = train_autoencoder(autoencoder_pokemon, pokemon_tensor, num_epochs=1000, batch_size=100)

    autoencoder_pokemon.eval()
    with torch.no_grad():
        latent_vectors, output_vectors = autoencoder_pokemon(pokemon_tensor)

    torch.save(autoencoder_pokemon, "autoencoder_pokemon.pth")
    print(f"Final Loss: {losses_pokemon[-1]}")

    return latent_vectors.cpu(), output_vectors.cpu()


if __name__ == "__main__":
    # latent_vectors, output_vectors = get_latent_pokemon() # or get_latent_moves
    latent_vectors, output_vectors = get_latent_moves()

    pyplot.hist(latent_vectors.numpy().flatten(), bins=50)
    pyplot.title("Latent Space Distribution")
    pyplot.show()
