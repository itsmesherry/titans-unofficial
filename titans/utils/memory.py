import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# The memory of a batch of sequences: (params, momentum).
# params holds, for each memory layer, a weight of shape (batch, out, in) and a
# bias of shape (batch, out), so every sequence has its own copy of the memory
# network. momentum has the same layout and holds the past surprise S_t.
MemoryState = Tuple[List[torch.Tensor], List[torch.Tensor]]


# Shifts the forgetting gate so that an untrained memory keeps about 98% of its
# weights per token. Centered on 0.5 it would halve them at every token, which
# wipes the memory within a few tokens and leaves its ReLU units dead.
FORGET_GATE_SHIFT = 4.0


def detach_state(state: MemoryState) -> MemoryState:
    """Cut a memory state off from the graph that produced it."""
    params, momentum = state
    return [p.detach() for p in params], [m.detach() for m in momentum]


class NeuralMemoryModule(nn.Module):
    """
    Neural Memory Module that learns to memorize at test time.
    
    Implements equations from Section 3.1 of the paper, using gradient descent
    with momentum and weight decay to update memory parameters during inference.
    
    The module itself holds no memory. Its parameters are the learned starting
    point of the memory network (memory_layers) and the projections and gates
    that drive the updates; the memory of a given batch of sequences lives in a
    MemoryState that is passed in and returned. Updates are differentiable, so
    every parameter can be trained through them.
    """
    
    def __init__(
        self,
        d_model: int,
        memory_depth: int = 2,
        memory_dim: Optional[int] = None,
        use_momentum: bool = True,
        use_weight_decay: bool = True,
        max_grad_norm: Optional[float] = 1.0,
    ):
        """
        Initialize the Neural Memory Module.
        
        max_grad_norm clips the norm of each sequence's surprise gradient. Plain
        gradient steps on the memory diverge once its error grows large; set it
        to None to use them anyway, as the paper's equations are written.
        """
        super().__init__()
        
        self.d_model = d_model
        self.memory_depth = memory_depth
        self.memory_dim = memory_dim if memory_dim is not None else d_model
        self.use_momentum = use_momentum
        self.use_weight_decay = use_weight_decay
        self.max_grad_norm = max_grad_norm
        
        # Projection matrices for key, value, and query
        self.W_K = nn.Linear(d_model, self.memory_dim, bias=False)
        self.W_V = nn.Linear(d_model, self.memory_dim, bias=False)
        self.W_Q = nn.Linear(d_model, self.memory_dim, bias=False)
        
        # Data-dependent parameters for surprise metric and forgetting
        self.theta_proj = nn.Linear(d_model, 1)  # Learning rate (θ)
        self.eta_proj = nn.Linear(d_model, 1)    # Momentum decay (η)
        self.alpha_proj = nn.Linear(d_model, 1)  # Weight decay (α)
        
        # The memory MLP. These weights are where every sequence's memory starts
        self.memory_layers = nn.ModuleList()
        for i in range(memory_depth):
            self.memory_layers.append(nn.Linear(self.memory_dim, self.memory_dim))

    def init_state(self, batch_size: int) -> MemoryState:
        """Return a fresh memory for each sequence, starting from the learned weights."""
        params = []
        for layer in self.memory_layers:
            params.append(layer.weight.unsqueeze(0).expand(batch_size, -1, -1))
            params.append(layer.bias.unsqueeze(0).expand(batch_size, -1))
        return params, [torch.zeros_like(p) for p in params]
    
    def retrieve(self, x: torch.Tensor, state: MemoryState) -> torch.Tensor:
        """
        Read from memory without changing it: M*(q) for every token.
        
        x has shape (batch, seq_len, d_model).
        """
        params, _ = state
        # Queries and keys are L2-normalized (Section 4.4)
        h = F.normalize(self.W_Q(x), dim=-1)
        for i in range(self.memory_depth):
            weight, bias = params[2 * i], params[2 * i + 1]
            h = torch.baddbmm(bias.unsqueeze(1), h, weight.transpose(1, 2))
            if i < self.memory_depth - 1:
                h = F.relu(h)
        return h
    
    def _surprise(
        self, k: torch.Tensor, v: torch.Tensor, params: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Gradient of the associative memory loss ||M(k) - v||^2 (momentary surprise).
        
        Returns one gradient per entry of params, separately for each sequence.
        Backpropagation is written out by hand so that the result is an ordinary
        differentiable expression: it works under torch.no_grad() and lets the
        outer optimizer train W_K and W_V through the memory updates.
        """
        # Forward pass, keeping each layer's input and ReLU mask
        inputs, masks = [], []
        h = k
        for i in range(self.memory_depth):
            inputs.append(h)
            h = torch.einsum('boi,bi->bo', params[2 * i], h) + params[2 * i + 1]
            if i < self.memory_depth - 1:
                masks.append(h > 0)
                h = F.relu(h)
        
        # Backward pass
        delta = 2 * (h - v)
        grads: List[torch.Tensor] = [delta] * (2 * self.memory_depth)
        for i in reversed(range(self.memory_depth)):
            grads[2 * i] = delta.unsqueeze(2) * inputs[i].unsqueeze(1)
            grads[2 * i + 1] = delta
            if i > 0:
                delta = torch.einsum('bo,boi->bi', delta, params[2 * i]) * masks[i - 1]
        return grads
    
    def _update_step(self, x_t: torch.Tensor, state: MemoryState) -> MemoryState:
        """Write one token of shape (batch, d_model) into memory."""
        params, momentum = state
        
        # Project input to key and value. Unit-norm keys also keep the gradient
        # step below stable: its size no longer grows with the input's norm
        k = F.normalize(self.W_K(x_t), dim=-1)
        v = self.W_V(x_t)
        
        # Compute data-dependent parameters
        theta = torch.sigmoid(self.theta_proj(x_t))
        eta = torch.sigmoid(self.eta_proj(x_t))
        alpha = torch.sigmoid(self.alpha_proj(x_t) - FORGET_GATE_SHIFT)
        
        grads = self._surprise(k, v, params)
        
        if self.max_grad_norm is not None:
            # One norm per sequence, over all memory parameters
            norm = torch.sqrt(sum(g.flatten(1).pow(2).sum(dim=1) for g in grads))
            scale = (self.max_grad_norm / (norm + 1e-6)).clamp(max=1.0)
            grads = [g * scale.view((-1,) + (1,) * (g.dim() - 1)) for g in grads]
        
        new_params, new_momentum = [], []
        for param, past_surprise, grad in zip(params, momentum, grads):
            # Gates are per sequence: (batch, 1) or (batch, 1, 1)
            shape = (-1,) + (1,) * (param.dim() - 1)
            
            # S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)
            surprise = -theta.view(shape) * grad
            if self.use_momentum:
                surprise = eta.view(shape) * past_surprise + surprise
            
            # M_t = (1 - α_t) * M_{t-1} + S_t
            if self.use_weight_decay:
                param = (1 - alpha.view(shape)) * param + surprise
            else:
                param = param + surprise
            
            new_params.append(param)
            new_momentum.append(surprise)
        return new_params, new_momentum
    
    def update(self, x: torch.Tensor, state: MemoryState) -> MemoryState:
        """Write a sequence of shape (batch, seq_len, d_model) into memory, token by token."""
        for t in range(x.size(1)):
            state = self._update_step(x[:, t], state)
        return state
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[MemoryState] = None,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, MemoryState]:
        """
        Read each token from memory, then write it.
        
        The output for token t comes from the memory as it was before token t
        was written, so it depends on the past only. With is_training=False the
        memory is read but not written. Returns the outputs and the new state;
        state=None starts from the learned memory.
        """
        batch_size, seq_len, _ = x.shape
        if state is None:
            state = self.init_state(batch_size)
        
        outputs = []
        for t in range(seq_len):
            # Retrieve from memory
            outputs.append(self.retrieve(x[:, t:t + 1], state))
            
            # Update memory parameters if in training mode
            if is_training:
                state = self._update_step(x[:, t], state)
        
        if not outputs:
            return x.new_zeros(batch_size, 0, self.memory_dim), state
        return torch.cat(outputs, dim=1), state
