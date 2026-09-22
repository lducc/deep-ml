import torch

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> torch.Tensor:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x) as a scalar torch.Tensor
    """
    def f(coeffs: list, x: float):
        n = len(coeffs)
        total = 0
        for i in range(n):
            total += coeffs[i] * x ** (n - i - 1)
        return total

    def f_(coeffs: list, x: float):
        n = len(coeffs)
        total = 0
        for i in range(n - 1):
            total += (n - i - 1) * coeffs[i] * x ** (n - i - 2)

        return total

    
    g = f(g_coeffs, x)
    gp = f_(g_coeffs, x)

    h = f(h_coeffs, x)
    hp = f_(h_coeffs, x)

    ans = torch.tensor((gp * h - g * hp) / h**2)
    return ans