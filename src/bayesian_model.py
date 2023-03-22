import numpy as np
import numpy.random as nprd
import numpy.linalg as nplg
from tqdm import tqdm
from scipy.stats import boltzmann

### Hyperparameters

tau2 = 1
r = 5  
beta_max = 100
max_neighbors = 80 # be sure that there are more data than max_neighbors

# ----------------------------- Model -------------------------------------

def pseudo_conditional(y, X, beta, k):
    """Compute the pseudo-conditional to plug-in the acceptance ratio
    of the Metropolis-Hastings algorithm

    Args:
        y (numpy.ndarray): labels 
        X (numpy.ndarray): data features (points)
        beta (float): temperature parameter
        k (int): number of neighbors to take into account

    Returns:
        float: the pseudo-conditionnal value
    """

    prod_dens = 1
    for x_i, label_i in zip(X, y):
        # k-nearest-neighbors of x_i
        dist_x_i = nplg.norm(X-x_i, axis=1)
        knn_xi_idx = dist_x_i.argsort()[:k+1] # takes itself into account
        nearest_labels_to_i = y[knn_xi_idx[1:]] # removes itself

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

def constant_ratio_approximation_mc(y, X, beta, k, mc_iter=100):
    """Estimate the normalization constant with Monte Carlo integration,
    to plug-in the acceptance ratio of the Metropolis-Hastings algorithm

    Args:
        y (numpy.ndarray): labels
        X (numpy.ndarray): data features
        beta (float): temperature parameter
        k (int): number of neighbors to take into accounty
        mc_iter (int, optional): Number of iterations for the Monte Carlo integration. Defaults to 100.
    """
    def summation_given_y(y):
        sum1 = 0
        for x_i, y_i in zip(X, y):
            # k-nearest-neighbors of x_i
            dist_x_i = nplg.norm(X-x_i, axis=1)
            knn_xi_idx = dist_x_i.argsort()[:k+1] # takes itself into account
            nearest_labels_to_i = y[knn_xi_idx[1:]] # removes itself
            sum1 += np.sum([1 for label in nearest_labels_to_i if label == y_i])/k
        return sum1

    size = len(y)
    nb_labels = len(np.unique(y))
    size_iteration = 10

    def summation_given_u(u):
        # Summing over all y's possible is too costy, we select some of them randomly instead
        sum_y = 0
        for y_sample in range(size_iteration):
            y_sample = nprd.randint(0, nb_labels, size)
            sum1 = summation_given_y(y_sample)
            sum_y += np.exp(u*sum1)
        return sum_y

    def expectation_given_u(u):
        # Summing over all y's possible is too costy, we select some of them randomly instead
        normalization = summation_given_u(u)
        sum_expect = 0
        for y_sample in range(size_iteration):
            y_sample = nprd.randint(0, nb_labels, size)
            sum1 = summation_given_y(y_sample)
            sum_expect += sum1 * np.exp(u*sum1)/normalization
        return sum_expect

    # Monte Carlo integration
    mc_sum = 0
    for iter in range(mc_iter):
        u = nprd.random()*beta
        mc_sum += beta*expectation_given_u(u)/mc_iter

    normalization = np.exp(len(y)*np.log(2)+mc_sum)
    density = np.exp(beta*summation_given_y(y))/normalization

    return density

def z_sampling(y, X, beta_hat, k_hat, beta, k):
    """Compute the value of denisty g, evaluated with a sample z
    from the Boltzmann distribution (Gibbs sampler), to plug-in the
    acceptance ratio of the Metropolis-Hastings algorithm

    Args:
        y (numpy.ndarray): labels
        X (numpy.ndarray): data features
        beta (float): temperature parameter
        k (int): number of neighbors to take into accounty
        k_hat (int): customed parameter for the g density
        beta (float): customed parameter for the g density
    """
    
    def summation_given_target(z):
        sum1 = 0
        for x_i, z_i in zip(X, z):
            # k-nearest-neighbors of x_i
            dist_x_i = nplg.norm(X-x_i, axis=1)
            knn_xi_idx = dist_x_i.argsort()[:k+1] # takes itself into account
            nearest_labels_to_i = z[knn_xi_idx[1:]] # removes itself
            sum1 += np.sum([1 for label in nearest_labels_to_i if label == z_i])/k
        return sum1

    nb_labels = len(np.unique(y))
    z = boltzmann.rvs(beta/k, N=nb_labels, size=len(y))

    # No need for normalization because of the ratio in MH algorithm
    g_dens = np.exp(beta_hat/k_hat*summation_given_target(z))

    return np.exp(beta/k*summation_given_target(y))*g_dens/np.exp(beta/k*summation_given_target(z))
    

def jacobian(theta):
    return np.exp(theta)/((1+np.exp(theta))**2)

def uniform_k(k_old, r=r):
    k_up = [i for i in range(k_old, k_old+r) if i <= max_neighbors]
    k_down = [i for i in range(k_old - r, k_old) if i > 0]
    k_new = nprd.choice(k_down+k_up)

    return k_new, len(k_down+k_up)

# ------------ Metropolis-Hastings algorithm ------------------

def acceptance_threshold(y, X, theta_new, theta_old, k_new, k_old, method):
    """Define the acceptance ratio for the Metropolis-Hastings algorithm, used
    to estimated the parameters beta and k for the Bayesian model

    Args:
        y (numpy.ndarray): labels
        X (numpy.ndarray): data features
        theta_new (float): current parameter (determine beta)
        theta_old (float): proposed parameter (to validate)
        k_new (int): proposed parameter (to validate)
        k_old (int): current parameter
        method (str): method used to estimated the ratio

    Returns:
        float: Acceptance ratio
    """

    beta_new = beta_max*np.exp(theta_new)/(1+np.exp(theta_new))
    beta_old = beta_max*np.exp(theta_old)/(1+np.exp(theta_old))

    if method =='pseudo_likelihood':
        model_conditionnal_up = pseudo_conditional(y,
                                                   X,
                                                   beta_new,
                                                   k_new)
        model_conditionnal_down = pseudo_conditional(y,
                                                     X,
                                                     beta_old,
                                                     k_old)

    elif method == 'path_sampling':
        model_conditionnal_up = constant_ratio_approximation_mc(y,
                                                                X,
                                                                beta_new,
                                                                k_new)
        model_conditionnal_down = constant_ratio_approximation_mc(y,
                                                                  X,
                                                                  beta_old,
                                                                  k_old)
    elif method == 'z_sampling':
        model_conditionnal_up = z_sampling(y,
                                           X,
                                           1.45,
                                           15,
                                           beta_new,
                                           k_new)
        
        model_conditionnal_down = z_sampling(y,
                                             X,
                                             1.45,
                                             15,
                                             beta_old,
                                             k_old)


    prob_knew = 1/uniform_k(k_old, r=r)[1]
    prob_kold = 1/uniform_k(k_new, r=r)[1]
    threshold = (model_conditionnal_up*jacobian(theta_new))*prob_kold\
                        /(model_conditionnal_down*jacobian(theta_old)*prob_knew)

    return threshold

def metropolis_hastings(y, X, niter, method):
    """Perform the Metropolis-Halgorithm for each given method and data

    Args:
        y (numpy.ndarray): labels
        X (numpy.ndarray): data features
        niter (int): number of iteration for the algorithm
        method (str): method used to estimated the ratio

    Returns:
        list: List of couple of parameters (beta, k) that have been accepted
    """


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

def proba_with_params(g, x_new, y, X, beta, k):
    """Estimate the conditionnal probability (before integration) 
    for a data point to belong to a class

    Args:
        g (int): potential class
        x_new (numpy.ndarray): new data point to assign
        y (numpy.ndarray): labels
        X (numpy.ndarray): data features
        beta (float): temperature parameter
        k (int): number of neighbors to take into account

    Returns:
        float: an evaluated probability for the data point to belong to the class g
        (conditional to the parameters)
    """
    dist_x_new = nplg.norm(X-x_new, axis=1)
    knn_xnew_idx = dist_x_new.argsort()[:k+1] # takes itself into account
    nearest_labels_to_new = y[knn_xnew_idx[1:]] # removes itself

    new_nearest_to_labels = []
    for x_l, label in zip(X, y):
        dist_x_l = nplg.norm(X-x_l, axis=1)
        knn_xl_idx = dist_x_l.argsort()[:k+1] # takes itself into account
        if dist_x_l[knn_xl_idx][-1] >= nplg.norm(x_l-x_new):
            new_nearest_to_labels.append(label)

    sum1 = np.sum([1 for label in nearest_labels_to_new if label == g]
                    + [1 for label in new_nearest_to_labels if label == g])

    normalization = np.sum([np.exp(beta/k*np.sum([1 for label in nearest_labels_to_new if label == g]))
                    * np.exp(beta/k*np.sum([1 for label in new_nearest_to_labels if label == g]))
                    for g in np.unique(y)])

    conditional_dens = np.exp(beta/k*sum1)/normalization

    return conditional_dens

def proba_class(g, x_new, y, X, param_list):
    """Estimate the full probability for a new data point to belong to a class,
    with a summation on the parameters that have been accepted by the Metropolis-Hastings algorithm

    Args:
        g (int): potential class
        x_new (numpy.ndarray): new data point
        y (numpy.ndarray): labels
        X (numpy.ndarray): data features
        param_list (list): list of accepted parameters issued from the Metropolis-Hastings algorithm

    Returns:
        float: an evaluated probability for the data point to belong to the class g
    """

    M = len(param_list)
    proba_sum = 0
    for params in param_list:
        proba_sum += proba_with_params(g, x_new, y, X, *params)

    proba_sum /= M
    return proba_sum

def sanity_check(X, y, param_list):
    """Compute the mean accuracy to check if the model sanity

    Args:
        X (numpy.ndarray): data features
        y (numpy.ndarray): labels
        param_list (list): list of accepted parameters issued from the Metropolis-Hastings algorithm

    Returns:
        NoneType: None (just print the information)
    """
    accuracy = 0
    idx = 0
    for x_i in tqdm(X):
        pred = np.argmax([proba_class(g, x_i, y, X, param_list) for g in np.unique(y)])
        if pred == y[idx]:
            accuracy += 1
        idx += 1
    
    accuracy /= len(y)
    print(f"Accuracy: {(100*accuracy):0.1f}%")
    return None