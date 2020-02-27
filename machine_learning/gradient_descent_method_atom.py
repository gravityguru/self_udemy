import numpy as np
import matplotlib.pyplot as plt

def draw(x1,x2):
    ln = plt.plot(x1, x2, '-')
    plt.pause(0.0001)
    ln[0].remove()

def sigmoid(score):
    return 1/(1+np.exp(-score))

def calculate_error(line_parameters, points, y):
    m = points.shape[0]
    p = sigmoid(points*line_parameters)
    cross_entropy = -(1/m)*(np.log(p).T*y+np.log(1-p).T*(1-y))
    return cross_entropy

def gradient_descent(line_parameters, points, y, alpha):
    m = points.shape[0]
    for i in range(5000):
        p = sigmoid(points*line_parameters)
        gradient = (points.T * (p-y))*(alpha / m)
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1 = np.array([points[:,0].min(), points[:,0].max()])
        x2 = -b/w2 +x1 * (-w1 / w2)
        print('x1=',x1)
        draw(x1, x2)

n_pts = 50
np.random.seed(0)
bias = np.ones(n_pts)
random_x1_values = np.random.normal(-15, 4, n_pts)
random_x2_values = np.random.normal(-16, 4, n_pts)
random_x3_values = np.random.normal(5, 4, n_pts)
random_x4_values = np.random.normal(7, 4, n_pts)
top_region = np.array([random_x1_values, random_x2_values, bias]).T
bottom_region = np.array([random_x3_values, random_x4_values, bias]).T
# w1 = -0.2 w2 = -0.35 b = 5
line_parameters = np.matrix ([np.zeros(3)]).T
#x1 =np.array([bottom_region[:,0].min(),top_region[:,0].max()])
#x2 =np.array([bottom_region[:,1].max(),top_region[:,1].min()])

#x2 = -b/w2 +x1 * (-w1 / w2)
#w1x1+w2x2+b




top_region = np.array([ top_region[:,0],  top_region[:,1], bias]).T
bottom_region = np.array([bottom_region[:,0],  bottom_region[:,1], bias]).T
all_points = np.vstack((top_region,bottom_region ))
_, ax = plt.subplots(figsize=(5, 5))
ax.scatter(top_region[:,0], top_region [:,1], color = 'r')
ax.scatter(bottom_region[:,0], bottom_region [:,1], color = 'b')



print(all_points.shape)
print(line_parameters.shape)

linear_combination = all_points * line_parameters


probabilities = sigmoid(linear_combination)


y= np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)


print(calculate_error(line_parameters, all_points, y))

gradient_descent(line_parameters, all_points, y, 0.05)
plt.show()
