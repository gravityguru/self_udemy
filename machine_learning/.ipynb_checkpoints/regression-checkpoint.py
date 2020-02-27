{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def draw(x1,x2):\n",
    "    ln = plt.plot(x1, x2)\n",
    "\n",
    "def sigmoid(score):\n",
    "    return 1/(1+np.exp(-score))\n",
    "\n",
    "def calculate_error(line_parameters, points, y):\n",
    "    m = points.shape[0]\n",
    "    p = sigmoid(points*line_parameters)\n",
    "    cross_entropy = -(1/m)*(np.log(p).T*y+np.log(1-p).T*(1-y))\n",
    "    return cross_entropy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[17.05620938,  8.09088848,  1.        ],\n",
       "       [11.60062883, 15.80035367,  1.        ],\n",
       "       [13.91495194, 11.39457117,  1.        ],\n",
       "       [18.9635728 , 11.58712459,  1.        ],\n",
       "       [17.47023196, 13.64239401,  1.        ]])"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "n_pts = 5\n",
    "np.random.seed(0)\n",
    "bias = np.ones(n_pts)\n",
    "random_x1_values = np.random.normal(10, 4, n_pts)\n",
    "random_x2_values = np.random.normal(12, 4, n_pts)\n",
    "random_x3_values = np.random.normal(5, 4, n_pts)\n",
    "random_x4_values = np.random.normal(7, 4, n_pts)\n",
    "top_region = np.array([random_x1_values, random_x2_values, bias]).T\n",
    "bottom_region = np.array([random_x3_values, random_x4_values, bias]).T\n",
    "w1 = -0.2\n",
    "w2 = -0.35\n",
    "b = 5\n",
    "line_parameters = np.matrix ([w1, w2, b]).T\n",
    "x1 =np.array([bottom_region[:,0].min(),top_region[:,0].max()])\n",
    "#x2 =np.array([bottom_region[:,1].max(),top_region[:,1].min()])\n",
    "\n",
    "x2 = -b/w2 +x1 * (-w1 / w2)\n",
    "#w1x1+w2x2+b\n",
    "\n",
    "\n",
    "top_region"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 5.57617428,  8.33469731,  1.        ],\n",
       "       [10.81709403, 12.97631629,  1.        ],\n",
       "       [ 8.0441509 ,  6.17936694,  1.        ],\n",
       "       [ 5.48670007,  8.25227081,  1.        ],\n",
       "       [ 6.77545293,  3.58361704,  1.        ]])"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bottom_region"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 5.48670007 18.9635728 ] [-0.27811432 -7.97918446]\n"
     ]
    }
   ],
   "source": [
    "print(x1,x2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "top_region = np.array([ top_region[:,0],  top_region[:,1], bias]).T\n",
    "bottom_region = np.array([bottom_region[:,0],  bottom_region[:,1], bias]).T\n",
    "all_points = np.vstack((top_region,bottom_region ))\n",
    "_, ax = plt.subplots(figsize=(5, 5))\n",
    "ax.scatter(top_region[:,0], top_region [:,1], color = 'r')\n",
    "ax.scatter(bottom_region[:,0], bottom_region [:,1], color = 'b')\n",
    "\n",
    "draw(x1, x2)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(10, 3)\n",
      "(3, 1)\n"
     ]
    }
   ],
   "source": [
    "print(all_points.shape)\n",
    "print(line_parameters.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "matrix([[7.24305284],\n",
       "        [8.85024955],\n",
       "        [7.7710903 ],\n",
       "        [8.84820817],\n",
       "        [9.26888429],\n",
       "        [5.03237892],\n",
       "        [7.70512951],\n",
       "        [4.77160861],\n",
       "        [4.9856348 ],\n",
       "        [3.60935655]])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "linear_combination = all_points * line_parameters\n",
    "linear_combination"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "matrix([[0.22390504],\n",
       "        [0.05466842],\n",
       "        [0.14540679],\n",
       "        [0.05477401],\n",
       "        [0.0366542 ],\n",
       "        [0.72464508],\n",
       "        [0.15379651],\n",
       "        [0.77353691],\n",
       "        [0.73387356],\n",
       "        [0.91611103]])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "probabilities = sigmoid(linear_combination)\n",
    "probabilities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.],\n",
       "       [0.],\n",
       "       [0.],\n",
       "       [0.],\n",
       "       [0.],\n",
       "       [1.],\n",
       "       [1.],\n",
       "       [1.],\n",
       "       [1.],\n",
       "       [1.]])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y= np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)\n",
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.34085201]]\n"
     ]
    }
   ],
   "source": [
    "print(calculate_error(line_parameters, all_points, y))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
