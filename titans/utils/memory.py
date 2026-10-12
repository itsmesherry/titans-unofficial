import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class NeuralMemoryModule(nn.Module):
    """
    Neural Memory Module that learns to memorize at test time.
    
    Implements equations from Section 3.1 of the paper, using gradient descent
    with momentum and weight decay to update memory parameters during inference.
    """
    
    def __init__(
        self,
        d_model: int,
        memory_depth: int = 2,
        memory_dim: Optional[int] = None,
        use_momentum: bool = True,
        use_weight_decay: bool = True,
    ):
        """Initialize the Neural Memory Module."""
        super().__init__()
        
        self.d_model = d_model
        self.memory_depth = memory_depth
        self.memory_dim = memory_dim if memory_dim is not None else d_model
        self.use_momentum = use_momentum
        self.use_weight_decay = use_weight_decay
        
        # Projection matrices for key, value, and query
        self.W_K = nn.Linear(d_model, self.memory_dim, bias=False)
        self.W_V = nn.Linear(d_model, self.memory_dim, bias=False)
        self.W_Q = nn.Linear(d_model, self.memory_dim, bias=False)
        
        # Data-dependent parameters for surprise metric and forgetting
        self.theta_proj = nn.Linear(d_model, 1)  # Learning rate (θ)
        self.eta_proj = nn.Linear(d_model, 1)    # Momentum decay (η)
        self.alpha_proj = nn.Linear(d_model, 1)  # Weight decay (α)
        
        # Initialize the memory MLP
        self.memory_layers = nn.ModuleList()
        for i in range(memory_depth):
            self.memory_layers.append(nn.Linear(self.memory_dim, self.memory_dim))
        
        # Initialize momentum buffers to None
        self.surprise_momentum = None
        
        self.reset_memory()

    def reset_memory(self) -> None:
        """Reset the memory state."""
        # Initialize memory parameters
        for layer in self.memory_layers:
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)
        
        # Initialize momentum to None so it will be created on the right device
        self.surprise_momentum = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def memory_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the memory MLP without weight updates."""
        for i, layer in enumerate(self.memory_layers):
            if i < self.memory_depth - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x
    
    def compute_loss(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute the associative memory loss (equation 3)."""
        pred_v = self.memory_forward(k)
        loss = F.mse_loss(pred_v, v, reduction='none')
        loss = loss.clamp(min=1e-10, max=1e10)  # For numerical stability
        return loss.mean(dim=-1, keepdim=True)
    
    def compute_gradient(self, k: torch.Tensor, v: torch.Tensor) -> List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Compute the gradient of the loss (momentary surprise in eq. 2)."""
        device = k.device
        try:
            with torch.enable_grad():
                # Create temporary layers with requires_grad=True
                temp_layers = nn.ModuleList()
                for layer in self.memory_layers:
                    temp_layer = nn.Linear(layer.in_features, layer.out_features, 
                                        bias=layer.bias is not None)
                    temp_layer.weight.data.copy_(layer.weight.data)
                    if layer.bias is not None:
                        temp_layer.bias.data.copy_(layer.bias.data)
                    # Ensure temp layer is on the same device as input
                    temp_layer = temp_layer.to(device)
                    temp_layers.append(temp_layer)
                
                # Forward pass
                x = k.detach().clone()
                for i, layer in enumerate(temp_layers):
                    if i < self.memory_depth - 1:
                        x = F.relu(layer(x))
                    else:
                        x = layer(x)
                
                # Compute loss and gradients
                loss = F.mse_loss(x, v.detach().clone())
                loss = loss.clamp(min=1e-10, max=1e10)
                loss.backward()
                
                # Extract gradients
                gradients = []
                for layer in temp_layers:
                    weight_grad = layer.weight.grad.clone() if layer.weight.grad is not None else None
                    bias_grad = layer.bias.grad.clone() if layer.bias is not None and layer.bias.grad is not None else None
                    gradients.append((weight_grad, bias_grad))
                
                # Clean up
                del temp_layers
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                return gradients
                
        except Exception as e:
            print(f"Error in gradient computation: {e}")
            return [(None, None) for _ in range(len(self.memory_layers))]
    
    def update_memory(
        self, 
        x: torch.Tensor, 
        theta: torch.Tensor, 
        eta: torch.Tensor, 
        alpha: torch.Tensor
    ) -> None:
        """Update memory parameters using the surprise metric (equation 4)."""
        device = x.device
        
        # Project input to key and value
        k = self.W_K(x)
        v = self.W_V(x)
        
        # Compute gradients
        gradients = self.compute_gradient(k, v)
        
        # Update memory parameters
        for i, ((weight_grad, bias_grad), layer) in enumerate(zip(gradients, self.memory_layers)):
            # Skip update if gradients are None
            if weight_grad is None:
                continue
                
            # Initialize momentum if not exists
            if self.surprise_momentum is None:
                self.surprise_momentum = []
                for memory_layer in self.memory_layers:
                    # Create momentum buffers on the same device as the model
                    weight_momentum = torch.zeros_like(memory_layer.weight, device=device)
                    bias_momentum = torch.zeros_like(memory_layer.bias, device=device) if memory_layer.bias is not None else None
                    self.surprise_momentum.append((weight_momentum, bias_momentum))
            
            # Update momentum (S_t in equation 4)
            if self.use_momentum:
                weight_momentum, bias_momentum = self.surprise_momentum[i]
                # Ensure tensors are on the right device
                weight_momentum = weight_momentum.to(device)
                if bias_momentum is not None:
                    bias_momentum = bias_momentum.to(device)
                
                theta_scalar = theta.mean().item()
                eta_scalar = eta.mean().item()
                alpha_scalar = alpha.mean().item()
                
                # S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)
                # Ensure weight_grad is on the same device as weight_momentum
                weight_grad = weight_grad.to(device)
                weight_momentum = eta_scalar * weight_momentum - theta_scalar * weight_grad
                
                if bias_grad is not None and bias_momentum is not None:
                    bias_grad = bias_grad.to(device)
                    bias_momentum = eta_scalar * bias_momentum - theta_scalar * bias_grad
                
                self.surprise_momentum[i] = (weight_momentum, bias_momentum)
            else:
                theta_scalar = theta.mean().item()
                weight_grad = weight_grad.to(device)
                weight_momentum = -theta_scalar * weight_grad
                
                bias_momentum = None
                if bias_grad is not None:
                    bias_grad = bias_grad.to(device)
                    bias_momentum = -theta_scalar * bias_grad
            
            # Update weights with weight decay (forgetting mechanism in equation 4)
            if self.use_weight_decay:
                alpha_scalar = alpha.mean().item()
                # M_t = (1 - α_t) * M_{t-1} + S_t
                # Ensure device consistency
                layer.weight.data = ((1 - alpha_scalar) * layer.weight.data + weight_momentum).to(device)
                
                if layer.bias is not None and bias_momentum is not None:
                    layer.bias.data = ((1 - alpha_scalar) * layer.bias.data + bias_momentum).to(device)
            else:
                layer.weight.data = (layer.weight.data + weight_momentum).to(device)
                
                if layer.bias is not None and bias_momentum is not None:
                    layer.bias.data = (layer.bias.data + bias_momentum).to(device)
    
    def forward(
        self, 
        x: torch.Tensor, 
        is_training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Neural Memory Module."""
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        outputs = []
        memory_outputs = []
        
        for t in range(seq_len):
            # Get current token
            x_t = x[:, t, :]
            
            # Project to query for retrieval
            q_t = self.W_Q(x_t)
            
            # Retrieve from memory
            memory_output = self.memory_forward(q_t)
            memory_outputs.append(memory_output)
            
            # Update memory parameters if in training mode
            if is_training:
                # Compute data-dependent parameters
                theta_t = torch.sigmoid(self.theta_proj(x_t))
                eta_t = torch.sigmoid(self.eta_proj(x_t))
                alpha_t = torch.sigmoid(self.alpha_proj(x_t))
                
                # Update memory
                self.update_memory(x_t, theta_t, eta_t, alpha_t)
            
            # Output is the memory output
            outputs.append(memory_output)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1).to(device)
        memory_outputs = torch.stack(memory_outputs, dim=1).to(device)
        
        return outputs, memory_outputs
