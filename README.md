# Linear Regression using Gradient Decent and Stochastic Gradient Decent

In this problem, we consider a simple linear regression model with a modified loss function and try to solve it with Gradient Descant (GD) and Stochastic Gradient Descant (SGD).

In general setting, the data has the form ```{(x1, y1), (x2, y2), . . . , (xn, yn)}``` where ```xi``` is the d-dimensional feature vector and ```yi``` is a real-valued target. For this regression problem, we will be using linear prediction ```w⊤xi``` with the objective function:

![Objective Function](https://github.com/mrigankdoshy/linear-regression-gd-sgd/blob/main/Function.png?raw=true)

Please note that we simply dropped the intercept term by adding a constant feature, which is always equal to ONE to simplify estimation of the “intercept” term.

## Gradient Descent

In this part, you are asked to optimize the above objective function using gradient descent and plot the function values over different iterations, which can be done using the python library ```matplotlib```.

To this end, in ```Problem5.py```, fill in the function 
```python 
bgd_l2(data, y, w, eta, delta, lam, num_iter)
``` 
where ```data``` is a two dimensional numpy array with each row being a feature vector, y is a one-dimensional numpy array of target values, w is a one-dimensional numpy array corresponding to the weight vector, eta is the learning rate, delta and lam are parameters of the objective function. This function should return new weight vector, history of the value of objective function after each iteration (python list).

Run this function for the following settings and plot the history of objective function (you should expect a monotonically decreasing function over iteration number):
- η = 0.05, δ = 0.1, λ = 0.001, num_iter=50 
- η = 0.1, δ = 0.01, λ = 0.001, num_iter=50 
- η = 0.1, δ = 0, λ = 0.001, num_iter=100 
- η = 0.1, δ = 0, λ = 0, num_iter=100

## Stochastic Gradient Descent

In ```Problem5.py``` fill in the function 
```python 
sgd_l2(data, y, w, eta, delta, lam, num_iter, i)
``` 
where ```data```, ```y```, ```w```, ```lam```, ```delta``` and ```num_iter``` are same as previous part. In this part, you should use ```η / √t``` as a learning rate, where ```t``` is the iteration number, starting from 1. The variable ```i``` is for testing the correctness of your function. If ```i``` set to −1 then you just need to apply the normal SGD (randomly select the data point), which runs for ```num_iter```, but if ```i``` set to something else (other than −1), your code only needs to compute SGD for that specific data point (in this case, the ```num_iter``` will be 1!).

Run this function for the settings below and plot the history of objective function: 
- η = 1, δ = 0.1, λ = 0.5,num_iter=800
- η = 1, δ = 0.01, λ = 0.1, num_iter=800
- η=1,δ=0,λ=0,num_iter=40
- η=1,δ=0,λ=0,num_iter=800
