import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    m_a = torch.as_tensor(a)
    m_b = torch.as_tensor(b)

    if m_a.size(1) != m_b.size(0):
        return torch.tensor(-1)
    return torch.matmul(m_a, m_b)
