import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')
# print(data)
# plt.scatter(data['SQUARE_FT'], data['Price'])
# plt.show()

def loss_function(m,b,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].SQUARE_FT
        y = points.iloc[i]['Price']
        total_error += (y - (m*x + b)) ** 2
    total_error = total_error / float(len(points))
    return total_error

def gradient_descent(points, starting_m, starting_b, learning_rate):
    m_gradient = 0
    b_gradient = 0
    n=len(points)
    for i in range(n):
        x = points.iloc[i]['SQUARE_FT']
        y = points.iloc[i]['Price']
        
        m_gradient += -(2/n) * x *(y - (starting_m * x + starting_b))
        b_gradient += -(2/n) * (y - (starting_m * x + starting_b))
    new_m = starting_m - (learning_rate * m_gradient)
    new_b = starting_b - (learning_rate * b_gradient)
    return [new_m, new_b]

m = 0
b = 0
learning_rate = 0.0000001
epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"epoch: {i}")
    m,b = gradient_descent(data, m, b, learning_rate)
print("m: {}, b: {}".format(m,b))
print("total error: {}".format(loss_function(m,b,data)))
plt.scatter(data['SQUARE_FT'], data['Price'], color = 'red')
plt.plot(data['SQUARE_FT'], m*data['SQUARE_FT'] + b, color = 'blue')
plt.show()