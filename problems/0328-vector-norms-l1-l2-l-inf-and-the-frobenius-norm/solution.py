import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    match norm_type:
        case 'l1':
            res = torch.linalg.vector_norm(arr.flatten(), ord=1)

        case 'l2':
            res = torch.linalg.vector_norm(arr.flatten(), ord=2)

        case 'linf':
            res = torch.linalg.vector_norm(arr.flatten(), ord=float('inf'))

        case 'frobenius':
            if arr.dim() != 2:
                raise ValueError(
                    'arr must be 2D for the Frobenius norm'
                )
            res = torch.linalg.matrix_norm(arr, ord='fro')

        case _:
            raise ValueError(
                'norm_type must be l1, l2, linf, or frobenius'
            )

    return res.item()
    pass
