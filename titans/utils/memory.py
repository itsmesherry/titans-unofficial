import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class NeuralMemoryModule(nn.Module):
    """
    Neural Memory Module as described in the paper "Learning to Memorize at Test Time".
    
    This module implements a deep neural network that learns to memorize at test time.
    It uses gradient descent with momentum and weight decay to update its parameters
    during inference.
    """
    
    def __init__(
        self,
        d_model: int,
        memory_depth: int = 2,
        memory_dim: Optional[int] = None,
        use_momentum: bool = True,
        use_weight_decay: bool = True,
    ):
        """
        Initialize the Neural Memory Module.
        
        Args:
            d_model: Dimension of the model
            memory_depth: Depth of the memory MLP (number of layers)
            memory_dim: Dimension of the memory (if None, uses d_model)
            use_momentum: Whether to use momentum in the surprise metric
            use_weight_decay: Whether to use weight decay (forgetting mechanism)
        """
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
        self.theta_proj = nn.Linear(d_model, 1) # Learning rate
        self.eta_proj = nn.Linear(d_model, 1)   # Momentum decay
        self.alpha_proj = nn.Linear(d_model, 1) # Weight decay
        
        # Initialize the memory MLP
        self.memory_layers = nn.ModuleList()
        for _ in range(memory_depth):
            self.memory_layers.append(nn.Linear(self.memory_dim, self.memory_dim))
                
        # Initialize memory state
        self.reset_memory()

    def reset_memory(self) -> None:
        """Reset the memory state and clear any stored gradients."""
        # Initialize memory parameters
        for layer in self.memory_layers:
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)
        
        # Initialize momentum
        self.surprise_momentum = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def memory_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the memory MLP without weight updates.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor after passing through the memory MLP
        """
        for i, layer in enumerate(self.memory_layers):
            if i < self.memory_depth - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x
    
    def compute_loss(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the associative memory loss.
        
        Args:
            k: Key tensor
            v: Value tensor
            
        Returns:
            Loss value
        """
        # Forward pass through memory without weight update
        pred_v = self.memory_forward(k)
        
        # Compute MSE loss with numerical stability
        loss = F.mse_loss(pred_v, v, reduction='none')
        loss = loss.clamp(min=1e-10, max=1e10)  # Prevent extreme values
        return loss.mean(dim=-1, keepdim=True)
    
    def compute_gradient(self, k: torch.Tensor, v: torch.Tensor) -> List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Compute the gradient of the loss with respect to the memory parameters.
        
        Args:
            k: Key tensor
            v: Value tensor
            
        Returns:
            List of (weight_grad, bias_grad) tuples for each layer
        """
        # Enable gradient computation
        try:
            with torch.enable_grad():
                # Create a copy of the memory layers with requires_grad=True
                temp_layers = nn.ModuleList()
                for layer in self.memory_layers:
                    temp_layer = nn.Linear(layer.in_features, layer.out_features, 
                                        bias=layer.bias is not None)
                    temp_layer.weight.data.copy_(layer.weight.data)
                    if layer.bias is not None:
                        temp_layer.bias.data.copy_(layer.bias.data)
                    temp_layers.append(temp_layer)
                
                # Forward pass through temporary layers
                x = k.detach().clone()  # Detach input from computation graph
                activations = []
                for i, layer in enumerate(temp_layers):
                    if i < self.memory_depth - 1:
                        x = F.relu(layer(x))
                    else:
                        x = layer(x)
                    activations.append(x)
                
                # Compute loss with numerical stability
                loss = F.mse_loss(x, v.detach().clone())
                loss = loss.clamp(min=1e-10, max=1e10)
                
                # Compute gradients
                loss.backward()
                
                # Extract gradients
                gradients = []
                for layer in temp_layers:
                    weight_grad = layer.weight.grad.clone() if layer.weight.grad is not None else None
                    bias_grad = layer.bias.grad.clone() if layer.bias is not None and layer.bias.grad is not None else None
                    gradients.append((weight_grad, bias_grad))
                
                # Clean up
                del temp_layers
                del activations
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
        """
        Update the memory parameters using the surprise metric.
        
        Args:
            x: Input tensor
            theta: Learning rate
            eta: Momentum decay
            alpha: Weight decay
        """
        # Project input to key and value
        k = self.W_K(x)
        v = self.W_V(x)
        
        # Compute gradients
        gradients = self.compute_gradient(k, v)
        
        # Update memory parameters with momentum and weight decay
        for i, ((weight_grad, bias_grad), layer) in enumerate(zip(gradients, self.memory_layers)):
            # Initialize momentum if not exists
            if self.surprise_momentum is None:
                self.surprise_momentum = []
                for _ in range(len(self.memory_layers)):
                    weight_momentum = torch.zeros_like(layer.weight)
                    bias_momentum = torch.zeros_like(layer.bias) if layer.bias is not None else None
                    self.surprise_momentum.append((weight_momentum, bias_momentum))
            
            # Update momentum (past surprise)
            if self.use_momentum:
                weight_momentum, bias_momentum = self.surprise_momentum[i]
                # Ensure proper broadcasting by using mean values of the parameters
                theta_scalar = theta.mean().item()
                eta_scalar = eta.mean().item()
                alpha_scalar = alpha.mean().item()
                
                weight_momentum = eta_scalar * weight_momentum - theta_scalar * weight_grad
                if bias_grad is not None and bias_momentum is not None:
                    bias_momentum = eta_scalar * bias_momentum - theta_scalar * bias_grad
                self.surprise_momentum[i] = (weight_momentum, bias_momentum)
            else:
                # Ensure proper broadcasting by using mean values of the parameters
                theta_scalar = theta.mean().item()
                weight_momentum = -theta_scalar * weight_grad
                bias_momentum = -theta_scalar * bias_grad if bias_grad is not None else None
            
            # Update weights with weight decay (forgetting mechanism)
            if self.use_weight_decay:
                # Ensure proper broadcasting by using mean values of the parameters
                alpha_scalar = alpha.mean().item()
                layer.weight.data = (1 - alpha_scalar) * layer.weight.data + weight_momentum
                if layer.bias is not None and bias_momentum is not None:
                    layer.bias.data = (1 - alpha_scalar) * layer.bias.data + bias_momentum
            else:
                layer.weight.data = layer.weight.data + weight_momentum
                if layer.bias is not None and bias_momentum is not None:
                    layer.bias.data = layer.bias.data + bias_momentum
    
    def forward(
        self, 
        x: torch.Tensor, 
        is_training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Neural Memory Module.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            is_training: Whether to update the memory parameters
            
        Returns:
            Tuple of (output, memory_output)
        """
        batch_size, seq_len, _ = x.shape
        
        # Process the sequence token by token
        outputs = []
        memory_outputs = []
        
        for t in range(seq_len):
            # Get current token
            x_t = x[:, t, :]
            
            # Project to query for memory retrieval
            q_t = self.W_Q(x_t)
            
            # Retrieve from memory
            memory_output = self.memory_forward(q_t)
            memory_outputs.append(memory_output)
            
            # Update memory parameters if in training mode
            if is_training:
                # Compute data-dependent parameters
                theta_t = torch.sigmoid(self.theta_proj(x_t))   # Learning rate
                eta_t = torch.sigmoid(self.eta_proj(x_t))       # Momentum decay
                alpha_t = torch.sigmoid(self.alpha_proj(x_t))
                
                # Update memory
                self.update_memory(x_t, theta_t, eta_t, alpha_t)
            
            # Combine with input
            output = memory_output
            outputs.append(output)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        memory_outputs = torch.stack(memory_outputs, dim=1)
        
        return outputs, memory_outputs
