def df(x ,a, b):
    return 2 * a * x + b
def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = x0
    for step in range(steps):
        grad = df(x, a, b)
        x_new = x - lr * grad

        x = x_new
    
    return x