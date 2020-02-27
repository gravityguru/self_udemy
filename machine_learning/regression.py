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
   "execution_count": null,
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
   "execution_count": 38,
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
     "execution_count": 38,
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
   "execution_count": 39,
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
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "matrix([[-1.24305284],\n",
       "        [-2.85024955],\n",
       "        [-1.7710903 ],\n",
       "        [-2.84820817],\n",
       "        [-3.26888429],\n",
       "        [ 0.96762108],\n",
       "        [-1.70512951],\n",
       "        [ 1.22839139],\n",
       "        [ 1.0143652 ],\n",
       "        [ 2.39064345]])"
      ]
     },
     "execution_count": 40,
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
   "execution_count": 41,
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
     "execution_count": 41,
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
   "execution_count": 42,
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
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y= np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)\n",
    "y\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
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
