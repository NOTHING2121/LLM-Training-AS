import torch

# True: mask (not attn)
# False: not mask (attn)

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Create a mask to identify non-padding positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
    Returns:
        A boolean mask tensor with shape (N, T), where: 
            - padding positions are marked with True 
            - non-padding positions are marked with False.
    """
    # TODO: Implement PadMask
    N, T = padded_input.shape[:2]  # 取前两维
    # [0, 1, 2, ..., T-1]
    time_idx = torch.arange(T, device=padded_input.device).repeat(N, 1)  # (1, T) -> (1*N, T*1)
    # [False, False, False, ..., True]
    mask = time_idx >= input_lengths.unsqueeze(1)  # (N, T) >= (N, 1) -> (N, T)
    return mask  # (N, T)

''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Create a mask to identify non-causal positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
    
    Returns:
        A boolean mask tensor with shape (T, T), where: 
            - non-causal positions (don't attend to) are marked with True 
            - causal positions (can attend to) are marked with False.
    """
    # TODO: Implement CausalMask
    T = padded_input.shape[1]

    # 创建上三角矩阵
    # 0: 主对角线（默认）, 1: 从主对角线上方开始, -1: 从主对角线下方开始
    mask = torch.triu(torch.ones((T, T), device=padded_input.device, dtype=torch.bool), diagonal=1)
    return mask  # (T, T)

