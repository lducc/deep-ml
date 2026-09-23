import torch

def descriptive_statistics(data) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset using PyTorch.
    
    Args:
        data: List, torch.Tensor, or array-like of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    data = torch.as_tensor(data, dtype=torch.float32)

    q1, median, q3 = data.quantile(torch.tensor([0.25, 0.5, 0.75]))

    mean = data.mean()
    mode = data.mode().values

    variance = data.var(correction=0)
    std = data.std(correction=0)

    iqr = q3 - q1

    return {
        'mean': mean.item(),
        'median': median.item(),
        'mode': mode.item(),
        'variance': round(variance.item(), 4),
        'standard_deviation': round(std.item(), 4),
        '25th_percentile': q1.item(),
        '50th_percentile': median.item(),
        '75th_percentile': q3.item(),
        'interquartile_range': iqr.item()
    }
