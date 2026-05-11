import torch
import math


def getCurrentLearningRateBasedOnSchedule (t, alpha_max, alpha_min, tw, tc):

    #Warm up:
    if t < tw:
        return (t/tw)*alpha_max
    
    #Cosine annealing
    if tw <= t <= tc: #as t goes from tw to tc, omega goes from 0 to pi
        omega = ((t-tw)/(tc - tw)) * math.pi
        return alpha_min + 0.5 * (1 + math.cos (omega)) * (alpha_max-alpha_min)

    #post annealing
    if t > tc:
        return alpha_min 

    assert False #unreachable!



#Profiling showed certain implementations to cause gpu2cpu async x-fer. notes to learn. now code keeps it in gpu!
def gradientClipping (params, max_l2norm, eps = 1e-6):
    l2_norm_sq= sum([(ele.grad**2).sum() for ele in params if ele.grad is not None]) #returns a tensor (1,). 
    #the list before sum() contains references to tensors on gpu. no d2h x-fer
    l2_norm = torch.sqrt (l2_norm_sq) 
    #Don't use math.sqrt() ; it causes a gpu2cpu transfer

    #l2_norm =  math.sqrt (sum([torch.norm(ele.grad.data).item()**2 for ele in params]))
    '''
    .item() in the second version moves each tensor to CPU as a Python float, which can be slightly slower on GPU.
    The first version keeps everything as PyTorch tensors, which is better to run fully on GPU.  
    '''


    '''
    #We want to avoid the "if" which compares torch scalar w/ cpu scalar causing a gpu2cpu x-fer

    if l2_norm > max_l2norm:
        #do clipping
        for param in params:
            if param.grad is None:
                continue
            scaling_factor = max_l2norm/(l2_norm + eps)
            #param.grad.data = param.grad.data * scaling_factor #isn't in-place!
            param.grad.data.mul_(scaling_factor)   #in-place
    '''

    scaling_factor = (max_l2norm/(l2_norm + eps)).clamp(max=1.0) #eliminates the "if" condition

    for param in params:
        if param.grad is None:
            continue
        param.grad.data.mul_(scaling_factor) #in-place!


class SGD (torch.optim.Optimizer):
    def __init__ (self, params, lr = 1e-3):

        assert lr > 0 , "learning rate > 0"

        hyperparams = {"lr" : lr}
        super().__init__(params, hyperparams)
    
    @torch.no_grad()
    def step (self, closure=None): 
        loss = None if closure is None else closure()

        for group in self.param_groups: #there are a set of groups
            lr = group ['lr'] #hyper-parameter for the group
            print (f"# OF Params in this param group = {len(group['params'])}")


            for param in group['params']: #for each param in this group
                if param.grad is None: #handles requires_grad=False; i.e., freezing!
                    continue 
                else:
                    param_state = self.state[param] #state assoc. w/ that param (a defaultdict)
                    t = param_state.get ('t', 0)
                    grad = param.grad.data
                    param.data = param.data - lr*grad/(math.sqrt (t+1))
                    param_state['t'] = t + 1
        
        return loss
    
class AdamW (torch.optim.Optimizer):
    def __init__ (self, params, lr = 1e-3, betas = (0.9, 0.999), eps=1e-8, weight_decay = 1e-2):

        beta1, beta2 = betas[0], betas[1]
        hyperparams = {"lr" : lr, "beta1" : beta1, "beta2" : beta2, "epsilon": eps, "weight_decay" : weight_decay}
        super().__init__(params, hyperparams)
     
    @torch.no_grad()
    def step (self, closure = None):
        loss = None if closure is None else closure()

        for group in self.param_groups: #there are a set of param groups

            #each group has a set of hyper-params
            lr = group ['lr'] #hyper-parameter for the group
            beta1 = group ['beta1']
            beta2 = group ['beta2']
            epsilon = group ['epsilon']
            weight_decay = group ['weight_decay']

            isDenominatorComputed = False

            #each group has a set of parameters
            for param in group['params']:
                if param.grad is None: #in-case: requires_grad is set to False!
                    continue

                #Get the state associated with each param
                param_state = self.state [param]
                grad = param.grad.data #get the gradient associated with the parameter

                assert param.grad.shape == param.data.shape

                m_t = param_state.get ('m_t', torch.zeros_like (param.data))
                v_t = param_state.get ('v_t', torch.zeros_like (param.data))
                t = param_state.get ("t", 1)

                m_t = beta1* m_t + (1-beta1) * grad
                v_t = beta2* v_t + (1-beta2) * (grad**2)


                if not isDenominatorComputed: #Do bias correction only once; since it is the same for all params!
                    beta1_pow_t = math.pow(beta1,t)
                    beta2_pow_t = math.pow(beta2,t)
                    isDenominatorComputed = True
                

                #bias correction 
                m_t_hat = m_t / (1-beta1_pow_t)
                v_t_hat = v_t / (1-beta2_pow_t)

                #Store the new state values that we need for next iteration!
                param_state['m_t'] = m_t #note that the vals stored are pre-bias correction!
                param_state['v_t'] = v_t
                param_state['t'] = t + 1 


                param.data = param.data - lr * (m_t_hat/ (torch.sqrt (v_t_hat) + epsilon) )
                
                #Apply weight decay
                param.data = param.data - lr*weight_decay*param.data 


                
        return loss


if __name__ == '__main__':
    torch.manual_seed(42)

    weights = torch.nn.Parameter (torch.randn(5,5)*5)
    bias = torch.nn.Parameter (torch.randn (5))
    optimizer = AdamW([weights, bias], lr = 1e1)

    for t in range(10):
        optimizer.zero_grad() #zeroes out the gradients
        loss = ((weights**2) + bias).mean()
        print (loss.cpu().item())

        loss.backward() #Computes the gradients
        gradientClipping ([weights,bias], 1e-2)
        optimizer.step() #Updates the weights

        #if we want to see the optimizer
        print (optimizer.state_dict())

