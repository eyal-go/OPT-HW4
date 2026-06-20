import numpy as np
import matplotlib.pyplot as plt

B = 0.5 * np.array([[3, 1], [1, 3]])

def ArmijoLinesearch(x, obj_f, grad_x, d, maxIter):
    #Initialize Armijo constants
    alpha = 1
    beta = 0.5
    c = 1e-4

    obj_x = obj_f(x)
    tangent_line = np.dot(grad_x, d)
    
    #Start linesearch
    for i in range(maxIter):
        curr_val = obj_f(x + alpha*d)
        if(curr_val <= obj_x + c*alpha*tangent_line):
            return alpha, i #Decreasing step size found
        else:
            alpha = beta*alpha
    
    return alpha, maxIter #if we got maxIter, we will terminate the search.
   
### Gradients and Hessians of g(x) ###
def g(x):
    x1 = x[0]
    x2 = x[1]

    return ((x1**2 + x2 -11)**2 + (x1 + x2**2 -7)**2)

def grad_g(x):
    x1 = x[0]
    x2 = x[1]

    dg_dx1 = 2*(x1**2 + x2 -11)*2*x1 + 2*(x1 + x2**2 - 7)
    dg_dx2 = 2*(x1**2 + x2 -11) + 2*(x1 + x2**2 - 7)*2*x2

    return np.array([dg_dx1, dg_dx2])

def hess_g(x):
    x1 = x[0]
    x2 = x[1]

    d2g_dx12 = 8*x1**2 + 4*(x1**2 + x2 -11) + 2
    d2g_dx22 = 8*x2**2 + 4*(x1 + x2**2 -7) + 2
    d2g_dx1x2 = 4*(x1 + x2)

    return np.array([[d2g_dx12, d2g_dx1x2], [d2g_dx1x2, d2g_dx22]])

### Gradient and Hessian of f(x) ###

def f(x):
    return g(B @ x)

def grad_f(x):
    return B @ grad_g(B @ x)

def hess_f(x):
    return B @ hess_g(B @ x) @ B


#Because GD doesn't use the actual Hessian of f, we don't need to worry about non-convexity.
def GradientDescent(x, maxIter, epsilon):
    #Initial values
    gradient_f = grad_f(x)

    #Initialize values and norms
    f_vals = [f(x)]
    f_norms = [np.linalg.norm(gradient_f)]
    for k in range(maxIter):
        x_norm = np.linalg.norm(x)
        d = gradient_f * -1
        
        alpha, iters = ArmijoLinesearch(x=x, obj_f=f, grad_x=gradient_f, d=d, maxIter=maxIter)
        x = x + alpha * d
        
        curr_f = f(x)
        gradient_f = grad_f(x)
        
        f_vals.append(curr_f)
        f_norms.append(np.linalg.norm(gradient_f))
        if(np.linalg.norm(alpha * d) / x_norm < epsilon):
            break
    return x, f_vals, f_norms

def Newton(x, maxIter, epsilon):
    #Initialize values
    gradient_f = grad_f(x)

    #Initialize values and norms
    f_vals = [f(x)]
    f_norms = [np.linalg.norm(gradient_f)]
    for k in range(maxIter):
        x_norm = np.linalg.norm(x)
        hessian_f = hess_f(x)
        inv_hess = np.linalg.inv(hessian_f)
        d = -1 *(inv_hess @ gradient_f)
        
        alpha, iters = ArmijoLinesearch(x=x, obj_f=f, grad_x=gradient_f, d=d, maxIter=maxIter)
        x = x + alpha * d
        
        curr_f = f(x)
        gradient_f = grad_f(x)
        
        f_vals.append(curr_f)
        f_norms.append(np.linalg.norm(gradient_f))
        if(np.linalg.norm(alpha * d) / x_norm < epsilon):
            break
    return x, f_vals, f_norms


x0 = np.random.randn(2)
x_sol, f_vals, f_norms = GradientDescent(x0, 100, 1e-6)

plt.semilogy(f_vals, color="Red", label="f(x) value")
plt.plot(f_norms, color="Blue", label="Gradient f(x) norm")
plt.title("Gradient Descent")
plt.legend()
plt.show()

x0 = np.random.randn(2)
x_sol, f_vals, f_norms = Newton(x0, 100, 1e-6)

plt.semilogy(f_vals, color="Red", label="f(x) value")
plt.plot(f_norms, color="Blue", label="Gradient f(x) norm")
plt.title("Newton's Method")
plt.legend()
plt.show()