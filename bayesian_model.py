import numpy as np 
import numpy.random as nprd
import numpy.linalg as nplg
import scipy as sp 

from tqdm import tqdm

### Hyperparameters

tau2 = 1
r = 5
beta_max = 4
max_neighbors = 40 # be sure that there are more data than max_neighbors

### Model 

def model_conditionnal(y, X, beta, k):
    return 1 

def pseudo_conditional(y, X, beta, k):
    
    prod_dens = 1
    for x_i, label_i in zip(X, y):
        # k-nearest-neighbors of x_i
        dist_x_i = nplg.norm(X-x_i, axis=1)
        knn_xi_idx = dist_x_i.argsort()[:k+1] # takes itself into account
        nearest_labels_to_i = y[knn_xi_idx[1:]]

        # x_l's for which x_i is one of the knn
        i_nearest_to_labels = []
        for x_l, label in zip(X, y):
            dist_x_l = nplg.norm(X-x_l, axis=1)
            knn_xl_idx = dist_x_l.argsort()[:k+1] # takes itself into account 
            if dist_x_l[knn_xl_idx][-1] >= nplg.norm(x_l-x_i):
                i_nearest_to_labels.append(label)

        sum1 = np.sum([1 for label in nearest_labels_to_i if label == label_i]
                      + [1 for label in i_nearest_to_labels if label == label_i])
        
        sum2 = np.sum([np.exp(beta/k*np.sum([1 for label in nearest_labels_to_i if label == g]))
                      * np.exp(beta/k*np.sum([1 for label in i_nearest_to_labels if label == g]))
                      for g in np.unique(y)])
        
        conditional_dens = np.exp(beta/k*sum1)/sum2
        prod_dens *= conditional_dens    

    return prod_dens

def constant_ratio_approximation_mc(y, X, beta, k, niter=100):
    def summation_given_y(y):
        sum1 = 0
        for x_i, y_i in zip(X, y):
            # k-nearest-neighbors of x_i
            dist_x_i = nplg.norm(X-x_i, axis=1)
            knn_xi_idx = dist_x_i.argsort()[:k+1] # takes itself into account
            nearest_labels_to_i = y[knn_xi_idx[1:]]
            sum1 += np.sum([1 for label in nearest_labels_to_i if label == y_i])/k
        return sum1
    
    nb_labels = len(np.unique(y))
    list_y = np.array(np.meshgrid(*[np.unique(y) for i in range(nb_labels)])).T.reshape(-1,nb_labels)
    
    def summation_given_u(u):
        sum_y = 0
        for y_sample in list_y:
            sum1 = summation_given_y(y_sample)
            sum_y += np.exp(u*sum1)
        return sum_y
    
    def expectation_given_u(u):
        normalization = summation_given_u(u)
        sum_expect = 0
        for y_sample in list_y:
            sum1 = summation_given_y(y_sample)
            sum_expect += sum1 * np.exp(u*sum1)/normalization
        return sum_expect
    
    # Monte Carlo integration
    mc_sum = 0
    for iter in range(niter):
        u = nprd.random()*beta
        mc_sum += beta*expectation_given_u(u)/niter

    normalization = np.exp(len(y)*np.log(2)+mc_sum)
    density = np.exp(beta*summation_given_y(y))/normalization
    
    return density


def jacobian(theta):
    return np.exp(theta)/((1+np.exp(theta))**2)

def uniform_k(k_old):
    k_up = [i for i in range(k_old, k_old+r) if i <= max_neighbors]
    k_down = [i for i in range(k_old - r, k_old) if i >= 0]
    k_new = nprd.choice(k_down+k_up)

    return k_new, len(k_down+k_up)

# ------------ Metropolis-Hastings algorithm ------------------

def acceptance_threshold(y, X, theta_new, theta_old, k_new, k_old, method):

    beta_new = beta_max*np.exp(theta_new)/(1+np.exp(theta_new))
    beta_old = beta_max*np.exp(theta_old)/(1+np.exp(theta_old))

    if method =='pseudo_likelihood':
        model_conditionnal_up = pseudo_conditional(y, X, beta_new, k_new)
        model_conditionnal_down = pseudo_conditional(y, X, beta_old, k_old)

    elif method == 'path_sampling':
        model_conditionnal_up = constant_ratio_approximation_mc(y, 
                                                                X, 
                                                                beta_new, 
                                                                k_new)
        model_conditionnal_down = constant_ratio_approximation_mc(y, 
                                                                  X, 
                                                                  beta_old, 
                                                                  k_old)
    
    prob_knew = 1/uniform_k(k_old)[1]
    prob_kold = 1/uniform_k(k_new)[1]
    threshold = (model_conditionnal_up*jacobian(theta_new))*prob_kold\
                        /(model_conditionnal_down*jacobian(theta_old)*prob_knew)

    return threshold

def metropolis_hastings(y, X, niter, method, burning=1000):

    if burning >= niter:
        burning = 0

    k_old = nprd.randint(max_neighbors+1)
    theta_old = nprd.normal(0, tau2)

    parameters =[]
    for iter in tqdm(range(niter)):
        k_new = uniform_k(k_old)[0]
        theta_new = nprd.normal(theta_old, tau2)
        u = nprd.random()
        
        acceptance = acceptance_threshold(y, 
                                          X, 
                                          theta_new, 
                                          theta_old, 
                                          k_new, 
                                          k_old, 
                                          method)
        if acceptance > u:
            theta_old = theta_new
            beta_old = beta_max*np.exp(theta_old)/(1+np.exp(theta_old))
            k_old = k_new
            parameters.append((beta_old, k_old))


    return parameters

# ------------- Probability of the label -------------


