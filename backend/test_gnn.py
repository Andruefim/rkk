import torch
from engine.causal_gnn import CausalGNNCore

def main():
    d = 10
    hidden = 24
    device = torch.device('cpu')
    print("Creating core...")
    core = CausalGNNCore(d, device, hidden)
    
    B = 2
    X = torch.randn(B, d)
    a = torch.zeros(B, d)
    
    print("Forward dynamics...")
    X_next = core.forward_dynamics(X, a)
    print(f"X_next shape: {X_next.shape}")
    
    print("Testing intervention loss...")
    loss = core.intervention_loss(X, X_next, 2, 1.0)
    print(f"Loss: {loss.item()}")
    
    print("Testing resize_to...")
    new_core = core.resize_to(15)
    print(f"New core d: {new_core.d}, mechanisms: {len(new_core.mechanisms)}")

if __name__ == "__main__":
    main()
