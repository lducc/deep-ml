import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    # Your code here
    if vectors is None or len(vectors) == 0:
        return True

    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    s = np.linalg.svd(vectors, compute_uv=False)

    return np.sum(s > 1e-10) == vectors.shape[0]