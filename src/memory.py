import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class NeuralMemoryModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        memory_depth: int = 2,
        memory_dim: Optional[int] = None,
        use_momentum: bool = True,
        use_weight_decay: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.memory_depth = memory_depth
        self.memory_dim = memory_dim if memory_dim is not None else d_model
        self.use_momentum = use_momentum
        self.use_weight_decay = use_weight_decay
        
        self.W_K = nn.Linear(d_model, self.memory_dim, bias=False)
        self.W_V = nn.Linear(d_model, self.memory_dim, bias=False)
        self.W_Q = nn.Linear(d_model, self.memory_dim, bias=False)
        
        self.theta_proj = nn.Linear(d_model, 1)
        self.eta_proj = nn.Linear(d_model, 1)
        self.alpha_proj = nn.Linear(d_model, 1)
        
        self.memory_layers = nn.ModuleList()
        for _ in range(memory_depth):
            self.memory_layers.append(nn.Linear(self.memory_dim, self.memory_dim))
                
        self.reset_memory()

    def reset_memory(self):
        for layer in self.memory_layers:
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)
        
        self.surprise_momentum = None
    
    def memory_forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.memory_layers):
            if i < self.memory_depth - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x
    
    def compute_loss(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        pred_v = self.memory_forward(k)
        
        loss = F.mse_loss(pred_v, v, reduction='none')
        return loss.mean(dim=-1, keepdim=True)
    
    def compute_gradient(self, k: torch.Tensor, v: torch.Tensor) -> list:
        with torch.enable_grad():
            temp_layers = nn.ModuleList()
            for layer in self.memory_layers:
                temp_layer = nn.Linear(layer.in_features, layer.out_features, 
                                      bias=layer.bias is not None)
                temp_layer.weight.data.copy_(layer.weight.data)
                if layer.bias is not None:
                    temp_layer.bias.data.copy_(layer.bias.data)
                temp_layers.append(temp_layer)
            
            x = k.detach().clone()
            activations = []
            for i, layer in enumerate(temp_layers):
                if i < self.memory_depth - 1:
                    x = F.relu(layer(x))
                else:
                    x = layer(x)
                activations.append(x)
            
            loss = F.mse_loss(x, v.detach().clone())
            
            loss.backward(retain_graph=True)
            
            gradients = []
            for layer in temp_layers:
                weight_grad = layer.weight.grad.clone() if layer.weight.grad is not None else None
                bias_grad = layer.bias.grad.clone() if layer.bias is not None and layer.bias.grad is not None else None
                gradients.append((weight_grad, bias_grad))
                
        return gradients
    
    def update_memory(
        self, 
        x: torch.Tensor, 
        theta: torch.Tensor, 
        eta: torch.Tensor, 
        alpha: torch.Tensor
    ) -> None:
        k = self.W_K(x)
        v = self.W_V(x)
        
        gradients = self.compute_gradient(k, v)
        
        for i, ((weight_grad, bias_grad), layer) in enumerate(zip(gradients, self.memory_layers)):
            if self.surprise_momentum is None:
                self.surprise_momentum = []
                for _ in range(len(self.memory_layers)):
                    weight_momentum = torch.zeros_like(layer.weight)
                    bias_momentum = torch.zeros_like(layer.bias) if layer.bias is not None else None
                    self.surprise_momentum.append((weight_momentum, bias_momentum))
            
            if self.use_momentum:
                weight_momentum, bias_momentum = self.surprise_momentum[i]
                theta_scalar = theta.mean().item()
                eta_scalar = eta.mean().item()
                alpha_scalar = alpha.mean().item()
                
                weight_momentum = eta_scalar * weight_momentum - theta_scalar * weight_grad
                if bias_grad is not None and bias_momentum is not None:
                    bias_momentum = eta_scalar * bias_momentum - theta_scalar * bias_grad
                self.surprise_momentum[i] = (weight_momentum, bias_momentum)
            else:
                theta_scalar = theta.mean().item()
                weight_momentum = -theta_scalar * weight_grad
                bias_momentum = -theta_scalar * bias_grad if bias_grad is not None else None
            
            if self.use_weight_decay:
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
        batch_size, seq_len, _ = x.shape
        
        outputs = []
        memory_outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            q_t = self.W_Q(x_t)
            
            memory_output = self.memory_forward(q_t)
            memory_outputs.append(memory_output)
            
            if is_training:
                theta_t = torch.sigmoid(self.theta_proj(x_t))
                eta_t = torch.sigmoid(self.eta_proj(x_t))
                alpha_t = torch.sigmoid(self.alpha_proj(x_t))
                
                self.update_memory(x_t, theta_t, eta_t, alpha_t)
            
            output = memory_output
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        memory_outputs = torch.stack(memory_outputs, dim=1)
        
        return outputs, memory_outputs
