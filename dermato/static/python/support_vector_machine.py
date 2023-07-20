import matplotlib.pyplot as plt
from matplotlib import style #(pour styliser le graph)
style.use('ggplot')
import numpy as np 

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)
            
    # entraînement :
    def fit(self, data):
        self.data = data
        
        # {||w|| : [w, b]}  
        optimization_dictionnary = {}
        
        transforms = [[1,1],
                        [-1,1],
                        [-1,-1],
                        [1,-1]]
        
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        
        step_sizes = [self.max_feature_value * 0.1,
                        self.max_feature_value * 0.01,
                        # point où cela devient coûteux:
                        self.max_feature_value * 0.001]
        # très coûteux:
        b_range_multiple = 5
        # lors de la descente de gradient les "pas" pour trouver 
        # le minimum global par rapport à b peuvent être plus grands 
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10
        
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # on peut optimiser car on a une courbe convexe
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple), self.max_feature_value * b_range_multiple, step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option  = True
                        # contrainte : yi * (xi*w + b) >= 1
                        # #### ajoiuter un break à la boucle for plus tard...
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    # break
                        if found_option:
                            optimization_dictionnary[np.linalg.norm(w_t)] = [w_t, b]
                            
                if w[0] < 0:
                    optimzed = True
                    print("l'un des pas est optimisé")
                else :
                    # w = [5,5]
                    # step = 1
                    # w - step = [4,4] ce qui est égal à w - [step, step]
                    w = w - step
            norms = sorted([ n for n in optimization_dictionnary])
            optimal_choice = optimization_dictionnary[norms[0]]
            # rappel: optimization_dictionnary = {||w|| : [w, b]}
            self.w = optimal_choice[0]
            self.b = optimal_choice[1]
            latest_optimum = optimal_choice[0][0] + step * 2
                                    
                                    
    def predict(self, features):
        # signe de (X.T * w + b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', color=self.colors[classification])
        return classification
        
    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=200, marker='*', color=self.colors[i]) for x in data_dictionnary[i]] for i in data_dictionnary]
        
        # forme de l'hyperplan: x*w + b
        # value = x*w + b
        # vecteur de support positif = 1
        # vecteur de support négatif = -1
        # frontière de décision = 0
        def hyperplane(x, w, b, value):
            return (-w[0] * x - b + value) / w[1]
            
        data_range = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyperplane_x_min = data_range[0]
        hyperplane_x_max = data_range[1]
        
        # (x*w + b) = 1
        # vecteur de support positif
        positive_support_vector1 = hyperplane(hyperplane_x_min, self.w, self.b, 1)
        positive_support_vector2 = hyperplane(hyperplane_x_max, self.w, self.b, 1)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max], [positive_support_vector1, positive_support_vector2])
        
        # (x*w + b) = -1
        # vecteur de support négatif
        negative_support_vector1 = hyperplane(hyperplane_x_min, self.w, self.b, -1)
        negative_support_vector2 = hyperplane(hyperplane_x_max, self.w, self.b, -1)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max], [negative_support_vector1, negative_support_vector2])
        
        # (x*w + b) = 0
        # frontière de décision
        decision_boundary1 = hyperplane(hyperplane_x_min, self.w, self.b, 0)
        decision_boundary2 = hyperplane(hyperplane_x_max, self.w, self.b, 0)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max], [decision_boundary1, decision_boundary2])
        
        plt.show()
    
data_dictionnary = {-1: np.array([[1,7],  # classe des -1
                                  [2,8],
                                  [3,8],]), 
                    1: np.array([[5,1],     # classe des 1
                                 [6,-1],
                                 [7,3],])}

svm = Support_Vector_Machine()
svm.fit(data = data_dictionnary)
svm.visualize()




