import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.special import gamma as gammafunc
from scipy.stats import gamma as gammadist
import pandas as pd
from tqdm.notebook import tqdm


class Bayesian_SIR:
    def __init__(self, initial_params, betas, gammas, distro='beta') -> None:
        self.N, self.T, self.I0 = initial_params
        self.betas = betas
        self.gammas = gammas 
        self.distro = distro    # to update beta

        # self.simulation()
        # self.run(...)

    """ Load data """

    def load_data(self, namefile='Singapore_new_smoothed.csv', N=0, T=0, do_plot=True, save=False):
        self.data = pd.read_csv(namefile)
        self.N = N if N else 5930134
        total_people = self.N
        self.T = T if T else 109

        self.data["Susceptible"] = total_people - np.cumsum(self.data["New_Infectious"])
        self.data["Infectious"] = np.cumsum(self.data["New_Infectious"] - self.data["New_Recovered"])
        self.data["Recovered"] = np.cumsum(self.data["New_Recovered"])

        self.configurations = np.zeros([self.T, 3])
        self.configurations[:, 0] = self.data["Susceptible"]
        self.configurations[:, 1] = self.data["Infectious"]
        self.configurations[:, 2] = self.data["Recovered"]

        if do_plot:
            xs = np.arange(0, self.T)
            ss = self.configurations[:, 0]/self.N
            ii = self.configurations[:, 1]/self.N + ss
            rr = self.configurations[:, 2]/self.N + ii

            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            ax[0].plot(ss, c = "black")
            ax[0].plot(ii, c = "black")
            ax[0].plot(rr, c = "black")
            ax[0].set_ylim([0, 1])

            ax[0].fill_between(xs, 0, ss, label = "S", color = "limegreen")
            ax[0].fill_between(xs, ss, ii, label = "I", color = "orangered")
            ax[0].fill_between(xs, ii, rr, label = "R", color = "cornflowerblue")

            ax[0].set_xlim([0, self.T-1])
            ax[0].set_xlabel('Day')
            ax[0].set_ylabel('Proportion')
            ax[0].legend()
            
            ax[1].plot(self.configurations[:, 0]/self.N, label = "S", color = "limegreen")
            ax[1].plot(self.configurations[:, 1]/self.N, label = "I", color = "orangered")
            ax[1].plot(self.configurations[:, 2]/self.N, label = "R", color = "cornflowerblue")
            ax[1].set_xlabel('Day')
            ax[1].grid(ls='--', alpha=0.5)
            ax[1].legend()

            plt.suptitle("Real data with varying parameters")
            if save:
                plt.savefig(f'real_data.png', dpi=500)
            plt.show()



    """ Data generation """

    def simulation(self, do_plot=True, save=False):
        initial_conf = np.array([self.N-self.I0, self.I0, 0])
        self.configurations = np.zeros([self.T, 3])
        self.configurations[0, :] = initial_conf

        for tt in range(1, self.T):
            beta_true = self.betas[tt]
            gamma_true = self.gammas[tt]
            
            delta_I = npr.binomial(self.configurations[tt-1][0],
                                   1 - np.exp(-beta_true*self.configurations[tt-1][1]/self.N))
            delta_R = npr.binomial(self.configurations[tt-1][1], gamma_true)
            
            self.configurations[tt][0] = self.configurations[tt-1][0] - delta_I
            self.configurations[tt][1] = self.configurations[tt-1][1] + delta_I - delta_R
            self.configurations[tt][2] = self.configurations[tt-1][2] + delta_R

        if do_plot:
            xs = np.arange(0, self.T)
            ss = self.configurations[:, 0]/self.N
            ii = self.configurations[:, 1]/self.N + ss
            rr = self.configurations[:, 2]/self.N + ii

            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            ax[0].plot(ss, c = "black")
            ax[0].plot(ii, c = "black")
            ax[0].plot(rr, c = "black")
            ax[0].set_ylim([0, 1])

            ax[0].fill_between(xs, 0, ss, label = "S", color = "limegreen")
            ax[0].fill_between(xs, ss, ii, label = "I", color = "orangered")
            ax[0].fill_between(xs, ii, rr, label = "R", color = "cornflowerblue")

            ax[0].set_xlim([0, self.T-1])
            ax[0].set_xlabel('Day')
            ax[0].set_ylabel('Proportion')
            ax[0].legend()
            
            ax[1].plot(self.configurations[:, 0]/self.N, label = "S", color = "limegreen")
            ax[1].plot(self.configurations[:, 1]/self.N, label = "I", color = "orangered")
            ax[1].plot(self.configurations[:, 2]/self.N, label = "R", color = "cornflowerblue")
            ax[1].set_xlabel('Day')
            ax[1].grid(ls='--', alpha=0.5)
            ax[1].legend()

            plt.suptitle("Simulated data with varying parameters")
            if save:
                plt.savefig(f'simulated_data.png', dpi=500)
            plt.show()


    def mylog(self, x):
        try:
            x = np.array(x)
            mask = x > 1e-323
            output = np.zeros(len(x))
            output[mask] = np.log(x[mask])
            output[~mask] = -744
            return output
        except:
            return np.log(x) if x > 0 else -744


    """ Delta update """

    def conditional_betagamma(self, delta, beta, gamma):
        K = np.sum(delta, dtype=int)
        eta = np.cumsum(delta)
        total = 0
        for ii in range(1, K+1):
            indic = eta == ii
            total = total + 0.2*np.log(0.1) - 2*np.log(gammafunc(0.1))
            total = total + 2*self.mylog(gammafunc(0.1 + np.sum(indic))) - np.sum(indic*self.mylog(gamma))
            total = total - (0.1+np.sum(indic))*(self.mylog(.1 + np.sum(indic*beta)) + self.mylog(0.1 - np.sum(indic*self.mylog(gamma))))
        return total

    def JJ(self, delta_proposed, delta_now, T):
        
        sum_proposed = np.sum(delta_proposed).astype(int)
        sum_now = np.sum(delta_now).astype(int)
        
        if sum_proposed == sum_now:
            return 1
        elif [sum_proposed, sum_now] == [1, 2] or [sum_proposed, sum_now] == [T, T-1]:
            return 3/(T-1)
        elif [sum_proposed, sum_now] == [2, 1] or [sum_proposed, sum_now] == [T-1, T]:
            return (T-1)/3
        elif sum_proposed < sum_now:
            return (sum_now - 1)/(T-sum_proposed)
        else:
            return (T- sum_now) / (sum_proposed - 1)
        
    def update_delta(self, delta_proposed, delta_now, beta_now, gamma_now, p, T):
        #Step 1: evaluate first term pi/pi
        difference = np.sum(delta_proposed - delta_now)
        first_term = difference*self.mylog(p/(1-p))
        
        #Step 2: evaluate second term:
        second_term = self.conditional_betagamma(delta_proposed, beta_now, gamma_now)
        second_term -= self.conditional_betagamma(delta_now, beta_now, gamma_now)
        
        #Step 3: evaluate third term
        third_term = self.mylog(self.JJ(delta_proposed, delta_now, T))
        
        log_m = first_term + second_term + third_term
            
        probability = min(0, log_m)
            
        eps = np.log(npr.uniform()) 
        if eps < probability:
            # print(f'Accepted move with p = {probability}')
            return delta_proposed
        else:
            # print(f'Not accepted move with p = {probability}')
            return delta_now
        
    def propose_delta(self, delta_now, T):
        delta = delta_now.copy()
        K = np.sum(delta, dtype=int)
        if K == 1:
            probs = np.array([1,0,0])
        elif K==T:
            probs = np.array([0,1,0])
        else:
            probs = np.array([1,1,1])/3
            
        choice = npr.choice([0,1,2], p = probs)
        
        if choice == 0: # add
            index = npr.choice(np.where(delta_now == 0)[0])
            # print(f'Adding in position {index}')
            delta[index] += 1
        elif choice == 1: # delete
            index = npr.choice(np.where(delta_now[1:] == 1)[0]) + 1
            # print(f'Deleting in position {index}')
            delta[index] -= 1
        else: # swap
            candidates = np.where((delta_now[1:T-1] - delta_now[2:T]) != 0)[0] + 1
            index_0 = npr.choice(candidates)
            # print(f'Swapping in position {candidates}, I choose: {index_0}')
            delta[index_0] = 1 - delta[index_0]
            delta[index_0+1] = 1 - delta[index_0+1]
        # print(f'Final delta = {delta}')
        return delta  
        

    """ b and r update """

    def update_b(self, delta_now, beta_now):
        eta = np.cumsum(delta_now)
        K = np.sum(delta_now)
        b_next = np.zeros(K)
        for kk in range(1, K+1):
            b_next[kk-1] = npr.gamma(0.1 + np.sum((eta==kk)), 1/(0.1+np.sum(beta_now*(eta==kk))))
        return b_next

    def update_r(self, delta_now, gamma_now):
        eta = np.cumsum(delta_now)
        K = np.sum(delta_now)
        r_next = np.zeros(K)
        for kk in range(1, K+1):
            r_next[kk-1] = npr.gamma(0.1 + np.sum((eta==kk)), 1/(0.1 + np.sum(-self.mylog(gamma_now)*(eta==kk))))
        return r_next
    

    """ β and γ update """
    
    # Sampling from a beta or exponential distribution 
    def update_beta_via_beta(self, S, I, N, T, b): #cool beta distribution trick
        new_betas = np.zeros(T)
        for tt in range(T):
            C = I[tt]/N
            D = -S[tt+1]+S[tt]
            A = C*(S[tt]-D)+b[tt]
            y = npr.beta(a = A/C, b = D+1)
            new_betas[tt] = -1/C*np.log(y)
        return new_betas

    # Sampling from a gamma distribution 
    def update_beta_via_gamma(self, S, I, N, T, b): 
        new_betas = np.zeros(T)
        for tt in range(T):
            delta_I = S[tt] - S[tt+1]
            P_t = I[tt]/N
            new_betas[tt] = npr.gamma(delta_I + 1, 1./(S[tt]-delta_I + b[tt]/P_t))/P_t
        return new_betas
    
    def update_beta(self, S, I, N, T, b, distro='beta'):
        return self.update_beta_via_gamma(S, I, N, T, b) if distro == ' gamma' else self.update_beta_via_beta(S, I, N, T, b)

    def update_gamma(self, I, R, N, T, r):
        new_gammas = np.zeros(T)
        for tt in range(T):
            delta_R = R[tt+1]-R[tt]
            new_gammas[tt] = npr.beta(delta_R + r[tt], I[tt]-delta_R+1)
        return new_gammas


    def run(self, p, n_steps, burnin, thin, data_file = "", N=0, T=0, do_plot=True, save=False):
        if not data_file:
            self.simulation(do_plot=False)
        else:
            self.load_data(data_file, N=N, T=T, do_plot=do_plot, save=save)

        T = self.T - 1
        betas_run = np.zeros([T, n_steps])
        gammas_run = np.zeros([T, n_steps])
        bs_run = np.zeros([T, n_steps])
        rs_run = np.zeros([T, n_steps])
        deltas_run = np.zeros([T, n_steps])

        # Initialize parameters
        delta_0 = (npr.uniform(size = T) < p).astype(int)
        delta_0[0] = 1

        KK = np.sum(delta_0)
        eta = np.cumsum(delta_0)
        rb = npr.gamma(shape = .1, scale = 10, size = (2, KK))
        r_0 = rb[0]
        b_0 = rb[1]

        r_0 = r_0[eta-1]
        b_0 = b_0[eta-1]

        beta_0 = npr.exponential(1/b_0)
        gamma_0 = npr.beta(r_0, 1)

        delta = delta_0.copy()
        beta = beta_0.copy()
        gamma = gamma_0.copy()
        r = r_0.copy()
        b = b_0.copy()

        print(f'Updating beta, gamma and delta parameters...')

        for step in tqdm(list(range(n_steps)), total=n_steps, desc='Buffering...', colour='green'):
            delta_new = self.propose_delta(delta, T)
            delta = self.update_delta(delta_new, delta, beta, gamma, p, T)

            b = self.update_b(delta, beta)
            r = self.update_r(delta, gamma)

            eta = np.cumsum(delta)
            b = b[eta-1]
            r = r[eta-1]
            
            beta = self.update_beta(self.configurations[:,0], self.configurations[:,1], self.N, T, b, distro=self.distro)
            gamma = self.update_gamma(self.configurations[:,1], self.configurations[:,2], self.N, T, r)
            
            if step >= burnin:
                betas_run[:, step] = beta
                gammas_run[:,step] = gamma
                deltas_run[:,step] = delta
                bs_run[:,step] = b
                rs_run[:,step] = r

        # Keep columns which are not empty
        self.deltas_samples = deltas_run[:, burnin::thin]
        self.bs_samples = bs_run[:, burnin::thin]
        self.rs_samples = rs_run[:, burnin::thin]
        self.betas_samples = betas_run[:, burnin::thin]
        self.gammas_samples = gammas_run[:, burnin::thin]
        
        self.etas_samples = np.cumsum(self.deltas_samples, axis=0)

        
    """ Find Bayes estimator δ^ """
    
    def find_Bayes_delta(self):
        T = self.T - 1

        # Compute q(t, t') matrix
        self.q_matrix = np.zeros(shape=(T, T))
        G = self.etas_samples.shape[1]
        for t in range(G):
            self.q_matrix += (self.etas_samples[:, t] == self.etas_samples[:, t][:, None]).astype(int)
        self.q_matrix = self.q_matrix/G
        
        # Find best delta
        print(f'Finding best delta minimizing the loss...')
        delta_final = np.array([0]*T)
        delta_final[0] = 1
        eta_hat = np.cumsum(delta_final)
        Index_add, Index_swap = [True, True]
        compute_loss = lambda eta, q_matrix: np.sum(np.abs((eta == eta[:, None]).astype(int) - q_matrix))

        current_loss = compute_loss(eta_hat, self.q_matrix)
        candidate_indexes = range(1, T)
        iteration = 0
        pbar = tqdm(total=100, desc='Loading...', colour='green')

        while(Index_add or Index_swap):
            iteration += 1
            # print(f'loss at iteration = {iteration}: {current_loss}')
            all_loss = []
            candidate_indexes = range(1, T)
            for i in candidate_indexes:
                delta_candidate = np.copy(delta_final)
                # Propose a transition:
                delta_candidate[i] = 1 - delta_candidate[i]
                eta_candidate = np.cumsum(delta_candidate)
                candidate_loss = compute_loss(eta_candidate, self.q_matrix)
                all_loss.append(candidate_loss)
            if min(all_loss) < current_loss:
                # better delta found: add transition
                current_loss = min(all_loss)
                best_index = candidate_indexes[np.argmin(all_loss)]
                delta_final[best_index] = 1 - delta_final[best_index]
                Index_add = True
            else:
                Index_add = False
            
            if np.sum(delta_final, dtype=int) in range(2, T):
                all_loss = []
                candidate_indexes = np.where((delta_final[1:T-1] - delta_final[2:T]) != 0)[0] + 1
                # print(candidate_indexes)
                for i in candidate_indexes:
                    delta_candidate = np.copy(delta_final)
                    # try different swaps
                    mask = np.array([0, 1]) + i
                    delta_candidate[mask] = 1 - delta_candidate[mask]

                    eta_candidate = np.cumsum(delta_candidate)
                    candidate_loss = compute_loss(eta_candidate, self.q_matrix)
                    all_loss.append(candidate_loss)
                if min(all_loss) < current_loss:
                    # better delta found: apply swap
                    current_loss = min(all_loss)
                    best_index = candidate_indexes[np.argmin(all_loss)]
                    best_index = np.array([0, 1]) + best_index
                    delta_final[best_index] = 1 - delta_final[best_index]
                    Index_swap = True
                else:
                    Index_swap = False
            pbar.update(1)
            if iteration > 100:
                # max iteration reached
                Index_add, Index_swap = [False, False]
        pbar.close()
        
        self.delta_final = delta_final

    """ Adjusted Rand Index """
    def ARI(self, delta_guess, delta_real):
        T = self.T - 1

        # Find eta distributions
        eta_guess = np.cumsum(delta_guess)
        eta_real  = np.cumsum(delta_real)

        # Loop on the distributions
        true_positives, false_positives, false_negatives, true_negatives = 0, 0, 0, 0

        for tt in range(T):
            for ttt in range(tt):
                
                ind_fun_real = eta_real[tt] == eta_real[ttt]
                ind_fun_guess = eta_guess[tt] == eta_guess[ttt]

                true_positives += ind_fun_real * ind_fun_guess * 2/(T*(T-1))
                false_positives += np.bitwise_not(ind_fun_real) * ind_fun_guess * 2/(T*(T-1))
                false_negatives += ind_fun_real * np.bitwise_not(ind_fun_guess) * 2/(T*(T-1))
                true_negatives += np.bitwise_not(ind_fun_real) * np.bitwise_not(ind_fun_guess) * 2/(T*(T-1))

        numer = (true_positives + true_negatives) - ((true_positives + false_positives)*(true_positives + false_negatives) +
                                                     (true_negatives + false_positives)*(true_negatives + false_negatives))
        denom = 1 - ((true_positives + false_positives)*(true_positives + false_negatives) + (true_negatives + false_positives)*(true_negatives + false_negatives))

        return numer/denom


    """ Mutual Information """
    # Mutual information
    def MI(self, delta_guess, delta_real):
        T = self.T - 1

        # Find the etas
        eta_real = np.cumsum(delta_real)
        eta_guess = np.cumsum(delta_guess)

        # Find the values for the sum
        K = sum(delta_real)
        K_hat = sum(delta_guess)

        # Find matrix n_k_k'

        # Loop to find information
        m_i = 0
        for kk in range(1, K+1):
            for kkk in range(1, K_hat+1):
                n_real = sum(eta_real == kk)
                n_guess = sum(eta_guess == kkk)
                nn = self.n_k_kp(kk, kkk, eta_real, eta_guess)
                m_i += nn/T * (self.mylog(nn * T) - self.mylog(n_guess*n_real))

        return m_i

    # Fid the matrix element for the mutual information
    def n_k_kp (self, kk, kkk, eta_real, eta_guess):
        nn = 0
        T = self.T - 1

        for ii in range(T):
            if eta_real[ii] == kk and eta_guess[ii] == kkk:
                nn += 1

        return nn
