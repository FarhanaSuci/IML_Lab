import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt

#GNN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GCN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

#generate simple graph
def create_sample_graph():
    #edges of the graph (source and target)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3, 3],  #source nodes
        [1, 0, 2, 1, 4, 5]   #target nodes
    ], dtype=torch.long)

    #node features(6 nodes, 3 features per node)
    x = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=torch.float)

    #node labels (for classification)
    y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

# train the GNN model
def train():
    #create graph data
    data = create_sample_graph()

    #define the model
    input_dim = data.x.size(1)
    hidden_dim = 16
    output_dim = 2  #number of classes
    model = GCN(input_dim, hidden_dim, output_dim)

    #define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    #training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    #evaluate model
    model.eval()
    _, pred = model(data).max(dim=1)

    # Plot actual vs. predicted
    plt.figure(figsize=(8, 6))
    plt.plot(data.y.numpy(), label="Actual", marker='o')
    plt.plot(pred.numpy(), label="Predicted", marker='x')
    plt.xlabel("Node Index")
    plt.ylabel("Class")
    plt.title("Actual vs Predicted Node Classes")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('actual_vs_predicted_node.png')
    print("actual_vs_predicted_node.png")

#run the training
if __name__ == "__main__":
    train()
