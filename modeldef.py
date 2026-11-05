#!/usr/bin/env python3 3.7.4 project3 env
# -*- coding: utf-8 -*-
"""
Toolbox for model definition.
"""
import numpy as np
import torch
import torch.nn as nn

# random
class RNN(nn.Module):
    '''Feedback of error: online position difference.'''
    def __init__(self, n_inputs, n_outputs, n_neurons, alpha, dtype, dt,
                 fwd_delay, fb_delay=0, biolearning=False, noiseout=0, noisein=0,
                 nonlin='relu', fb_sparsity=1, noise_kernel_size=1, rec_sparsity=1):
        super(RNN, self).__init__()
        self.n_neurons = n_neurons                # 400 units neurons
        self.n_inputs = n_inputs                  # Input dimension
        self.n_outputs = n_outputs                # Output dimension
        self.alpha = alpha                        # Time constant-related parameter (dt/tau) = 0.01/0.05 = 0.2
        self.dt = dt                              # Time step = 0.01s
        
        self.rnn = nn.RNN(n_inputs, n_neurons, num_layers=2, bias=False)     # Two-layer RNN
        self.output = nn.Linear(n_neurons, n_outputs)                        # Output layer: generates velocity
        self.feedback = nn.Linear(n_outputs, n_neurons)                      # Feedback layer: returns error to network
        self.dtype = dtype
        
        self.fwd_delay = fwd_delay            # simulates neural transmission delay
        self.fb_delay = fb_delay              # simulates sensory feedback delay
        self.delay = fwd_delay+fb_delay
        
        self.biolearning = biolearning
        if biolearning:
            self.rnn.weight_hh_l0.requires_grad = False     # The weights of the hidden-to-hidden connection are not updated, because the gradient is not computed.
        self.noisein = noisein
        self.noiseout = noiseout
        self.noise_kernel_lenk = noise_kernel_size
        
        if nonlin=='relu':
            self.nonlin = torch.nn.ReLU()
        elif nonlin=='tanh':
            self.nonlin = torch.nn.Tanh()
        elif nonlin=='sigmoid':
            self.nonlin = torch.nn.Sigmoid()

        self.mask = nn.Linear(n_outputs, n_neurons, bias=False)     # Sparse feedback connections, default fully connected
        self.mask.weight = nn.Parameter((torch.rand(n_neurons, n_outputs) < fb_sparsity).float()).type(dtype)
        self.mask.weight.requires_grad = False

        self.mask2 = nn.Linear(n_neurons, n_neurons, bias=False)    # Sparse recurrent connections, default fully connected
        self.mask2.weight = nn.Parameter((torch.rand(n_neurons, n_neurons) < rec_sparsity).float()).type(dtype)
        self.mask2.weight.requires_grad = False

        self.register_buffer('alive_mask', torch.ones(1, self.n_neurons))   # Alive mask for alive neurons when applying lesion, shape [1, N]
        self.p_dead_current = 0.0

        self.neuron_positions = torch.rand(self.n_neurons, 2)
        self.register_buffer('neuron_positions_buffer', self.neuron_positions)

        self.stim_weights = None
        self.stim_enabled = False

    def init_hidden(self):
        return ((torch.rand(self.batch_size, self.n_neurons)-0.5)*0.2).type(self.dtype)    # Initialization of hidden states, random numbers uniformly between [-0.1, 0.1] (different from the paper, might be a mistake)
    
    # ONE SIMULATION STEP
    def f_step(self,xin,x1,r1,v1fb,v1,pin,popto,stim_intensity=None, non_lin = True):

        # Fixed stimulation input (if enabled)
        stim_input = 0
        if self.stim_enabled and stim_intensity is not None and self.stim_weights is not None:
            # stim_intensity can be scalar or [batch, n_channels]
            if stim_intensity.dim() == 0:  # scalar
                stim_input = self.stim_weights * stim_intensity
            elif stim_intensity.dim() == 1:  # [batch]
                stim_input = self.stim_weights.unsqueeze(0) * stim_intensity.unsqueeze(1)
            else:  # [batch, time] etc.
                stim_input = self.stim_weights * stim_intensity.squeeze()

        x1 = x1 + self.alpha*(-x1 + r1 @ (self.mask2.weight*self.rnn.weight_hh_l0).T           # Recurrent input
                                  + xin[:,:3] @ self.rnn.weight_ih_l0.T                        # External input
                                  + v1fb @ (self.mask.weight*self.feedback.weight).T           # Feedback input
                                  + self.feedback.bias.T                                       # Bias term
                                  + popto                                                      # Microstimulation in the original experiment
                                  + stim_input                                                 # Microstimulation in my simulation
                              )
        # Apply the nonlinearity only if BOTH the per-call flag and the
        # model-level switch allow it.  Setting model.nonlinear=False runs the
        # whole network linearly (r1 = x1) through every f_step call, including
        # those made inside forward(), without threading a flag through the API.
        # getattr keeps older checkpoints/instances (no .nonlinear) working.
        if non_lin and getattr(self, 'nonlinear', True):
            r1 = self.nonlin(x1)
        else:
            r1 = x1                         # linear mode: firing rate = membrane potential
        r1 = r1 * self.alive_mask                      # Apply alive mask to keep firing rate of lesioned neurons at 0
        vt = self.output(r1) + pin                     # Velocity output (plus perturbation)
        v1 = v1 + self.dt*(xin[:,3:5]-vt)              # Position error integration
        return x1,r1,v1
    
    # BIOLOGICALLY PLAUSIBLE LEARNING RULE
    def dW(self,pre,post,lr,inp,fb,presum):
        with torch.no_grad():           # Not to calculate and update gradients
            return self.alpha*0.1*(
                + lr*self.mask2.weight*torch.outer(fb@(self.mask.weight*self.feedback.weight).T,presum)
                )
                    
    # GET VELOCITY OUTPUT (NOT ERROR)
    def get_output(self,testl1):
        if self.noiseout>0:
            return self.output(testl1)+self.output(testl1)*self.create_noise(testl1)     # Add signal-dependent noise
        else:
            return self.output(testl1)
        
    # SIMULATE MOTOR NOISE (scales with velocity output)
    def create_noise(self,testl1):
        # time varying noise
        tmp = self.noiseout*torch.randn(testl1.shape[1],testl1.shape[0],2)
        tmp = tmp.permute(0,2,1)
        lenk = self.noise_kernel_lenk
        kernel = torch.ones(2,tmp.shape[1],lenk)/lenk
        padding = lenk // 2 + lenk % 2
        noise = torch.nn.functional.conv1d(tmp, kernel, padding=padding)[:,:,:testl1.shape[0]]
        noise = noise.permute(2,0,1)*np.sqrt(lenk)  
        return noise

    # Lesion
    def set_dead_fraction(self, perm: torch.LongTensor, p_dead: float):
        N = self.n_neurons
        assert perm.numel() == N, "perm must be length N"
        perm = perm.to(self.alive_mask.device)

        k = int(N* p_dead)
        idx = perm[:k]
        m = torch.ones(N, device=self.alive_mask.device)
        if k > 0:
            m[idx] = 0.0
        self.alive_mask.copy_(m.view(1, -1))

    # STIMULATION
    def generate_random_neuron_positions(self):
        """
        Generate random 2D coordinates for neurons (not on grid).
        Note: Positions are already generated in __init__.
        This method is kept for backward compatibility.
        """
        if self.neuron_positions is None:
            self.neuron_positions = self.neuron_positions_buffer
        return self.neuron_positions

    def compute_distance_to_stimulation_sites(self, stimulation_sites):
        """
        Compute Euclidean distance from each neuron to each stimulation site.

        Args:
            stimulation_sites: [n_sites, 2] tensor of site coordinates in [0, 1]

        Returns:
            distances: [n_neurons, n_sites] tensor of distances
        """
        if self.neuron_positions is None:
            raise ValueError("Must generate neuron positions first!")

        # Calculate Euclidean distance
        # neuron_positions: [N, 2], stimulation_sites: [S, 2]
        # Result: [N, S]
        distances = torch.cdist(self.neuron_positions, stimulation_sites)

        return distances

    def distance_to_weight_reciprocal(self, distances, max_distance=0.1, power=1):
        """
        Convert distances to weights using reciprocal function.

        weight = 1 / (1 + (distance / scale)^power)

        Args:
            distances: [n_neurons, n_sites] distance matrix
            max_distance: Distance at which weight becomes ~0.5 (scaling factor)
            power: Exponent for distance scaling (higher = sharper drop-off)

        Returns:
            weights: [n_neurons, n_sites] weight matrix
        """
        # Reciprocal function: weight = 1 / (1 + (distance/max_distance)^power)
        normalized_dist = distances / max_distance
        weights = 1 / (1 + normalized_dist ** power)

        return weights

    def distance_to_weight_gaussian(self, distances, sigma=0.2):
        """
        Convert distances to weights using Gaussian function.

        weight = exp(-distance^2 / (2 * sigma^2))

        Args:
            distances: [n_neurons, n_sites] distance matrix
            sigma: Standard deviation (spread of stimulation)

        Returns:
            weights: [n_neurons, n_sites] weight matrix
        """
        weights = torch.exp(-distances ** 2 / (2 * sigma ** 2))
        return weights

    def distance_to_weight_linear(self, distances, max_distance=0.7):
        """
        Convert distances to weights using linear falloff.

        weight = max(0, 1 - distance / max_distance)

        Args:
            distances: [n_neurons, n_sites] distance matrix
            max_distance: Distance at which weight becomes 0

        Returns:
            weights: [n_neurons, n_sites] weight matrix
        """
        weights = torch.clamp(1 - distances / max_distance, min=0)
        return weights

    def set_stim_weights_from_sites(self, stimulation_sites,
                                    transform='reciprocal',
                                    combine='lin',
                                    **transform_kwargs):
        """
        Set stimulation weights based on distance to stimulation sites.

        Args:
            stimulation_sites: List or tensor of site coordinates [[x1,y1], [x2,y2], ...]
            transform: 'reciprocal', 'gaussian', or 'linear'
            combine: 'max', 'sum', or 'mean' (how to combine multiple sites)
            **transform_kwargs: Parameters for the transform function

        Returns:
            stim_weights: [n_neurons] final stimulation weights
        """
        # Convert to tensor if list
        if isinstance(stimulation_sites, list):
            stimulation_sites = torch.tensor(stimulation_sites, dtype=torch.float32)

        n_sites = stimulation_sites.shape[0]

        # Compute distances to all sites
        distances = self.compute_distance_to_stimulation_sites(stimulation_sites)

        # Convert distances to weights
        if transform == 'reciprocal':
            max_distance = transform_kwargs.get('max_distance', 0.03)
            power = transform_kwargs.get('power', 1)
            site_weights = self.distance_to_weight_reciprocal(distances, max_distance, power)
        elif transform == 'gaussian':
            sigma = transform_kwargs.get('sigma', 0.2)
            site_weights = self.distance_to_weight_gaussian(distances, sigma)
        elif transform == 'linear':
            max_distance = transform_kwargs.get('max_distance', 0.5)
            site_weights = self.distance_to_weight_linear(distances, max_distance)
        else:
            raise ValueError(f"Unknown transform: {transform}")

        # Combine weights from multiple sites
        if combine == 'max':
            stim_weights = site_weights.max(dim=1)[0]  # Take maximum weight
        elif combine == 'lin':
            stim_weights = site_weights.sum(dim=1)     # Linear combination of weights, can be greater than 1
        elif combine == 'sum':
            stim_weights = site_weights.sum(dim=1)
            # Normalize to keep weights in reasonable range
            stim_weights = torch.clamp(stim_weights, max=1.0)
        elif combine == 'mean':
            stim_weights = site_weights.mean(dim=1)
        else:
            raise ValueError(f"Unknown combine method: {combine}")

        # Store as fixed weights
        self.stim_weights = stim_weights
        self.stim_enabled = True
        self.stimulation_sites = stimulation_sites

        return stim_weights

    def set_single_site_center(self, center=(0.5, 0.5), transform='reciprocal', **kwargs):
        """
        Convenience method: set single stimulation site at center.

        Args:
            center: (x, y) coordinates in [0, 1]
            transform: 'reciprocal', 'gaussian', or 'linear'
            **kwargs: Parameters for transform
        """
        stimulation_sites = torch.tensor([center], dtype=torch.float32)
        return self.set_stim_weights_from_sites(stimulation_sites, transform, **kwargs)

    def set_random_sites(self, n_sites=1, transform='reciprocal', **kwargs):
        """
        Convenience method: set random stimulation sites.

        Args:
            n_sites: Number of random stimulation sites
            transform: 'reciprocal', 'gaussian', or 'linear'
            **kwargs: Parameters for transform
        """
        stimulation_sites = torch.rand(n_sites, 2)
        return self.set_stim_weights_from_sites(stimulation_sites, transform, **kwargs)

    def visualize_stimulation_profile(self, save_path=None):
        """
        Visualize the stimulation weight profile on 2D grid.
        """
        if self.neuron_positions is None or self.stim_weights is None:
            print("No neuron positions or stimulation weights set!")
            return

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        positions = self.neuron_positions.cpu().numpy()
        weights = self.stim_weights.cpu().numpy()

        scatter = axes[0].scatter(positions[:, 0], positions[:, 1],
                             c=weights, cmap='hot', s=20, alpha=0.6)
        axes[0].set_xlabel('X Position')
        axes[0].set_ylabel('Y Position')
        axes[0].set_title('Stimulation Weight Profile')
        plt.colorbar(scatter, label='Weight')

        if hasattr(self, 'stimulation_sites'):
            sites = self.stimulation_sites.cpu().numpy()
            axes[0].scatter(sites[:, 0], sites[:, 1],
                       c='black', s=200, marker='x')

        # Plot histogram
        axes[1].hist(self.stim_weights.cpu().numpy(), bins=50, alpha=0.7)
        axes[1].set_xlabel('Stimulation Weight')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Weight Distribution\nMean: {self.stim_weights.mean():.3f}\nMedian: {self.stim_weights.median():.3f}')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close(fig)


    # RUN MODEL
    def forward(self, X, Xpert, lr=1e-3, popto=None, stim_intensity=None):          # shape of X: [tsteps, batch_size, input_dimentions]
        self.batch_size = X.size(1)
        hidden0 = self.init_hidden()
        hidden0 = hidden0 * self.alive_mask
        x1 = hidden0
        r1 = self.nonlin(x1)
        r1 = r1 * self.alive_mask
        v1 = self.output(r1)
        x = []
        hidden1 = []
        poserr = []
        presum = r1
        dw_acc = torch.zeros(self.rnn.weight_hh_l0.shape)

        if popto is None:
            popto = torch.zeros((X.shape[0],X.shape[1],r1.shape[1]))

        # prerun simulation (until 'delay' is reached)
        for j in range(self.fwd_delay+1):
            stim_t = stim_intensity[j] if stim_intensity is not None else None
            x1,r1,v1 = self.f_step(X[0],x1,r1,
                                         v1*0,v1,Xpert[0],
                                         popto[0],stim_t)
            x.append(x1)
            hidden1.append(r1)
            poserr.append(v1)
        # generate noise (input and output)
        X = X+ self.noisein*torch.randn(X.shape[1],X.shape[2])[None,:,:]
        if self.noiseout>0:
            noise = self.create_noise(Xpert)
        else:
            noise = torch.zeros(Xpert.shape)
        # now real time simulation
        for j in range(X.size(0)):
            stim_t = stim_intensity[j] if stim_intensity is not None else None
            if (self.fb_delay<0) | (j<=self.fb_delay):     # without feedback by v1*0
                x1,r1,v1 = self.f_step(X[j],x1,r1,
                                       v1*0,v1,
                                       Xpert[j]+noise[j]*self.output(r1),
                                       popto[j],stim_t)
            else:
                x1,r1,v1 = self.f_step(X[j],x1,r1,
                                             poserr[j-self.delay],          # Delayed error signal
                                             v1,Xpert[j]+noise[j]*self.output(r1),
                                             popto[j],stim_t)
                if self.biolearning and j%5:
                    dw_acc += self.dW(x1[0],r1[0],lr,X[j,0],
                                      poserr[j-self.delay][0],presum[0]).detach()
            if self.biolearning:
                presum += r1.detach()           # A newly created tensor r1.detach() is added to presum. This tensor will never require gradient.
            x.append(x1)
            hidden1.append(r1)
            poserr.append(v1)
        x = torch.stack(x[1:])
        hidden1 = torch.stack(hidden1[1:])
        poserr = torch.stack(poserr[1:])
        if self.biolearning:
            with torch.no_grad():
                self.rnn.weight_hh_l0 += dw_acc
        if self.fwd_delay==0:
            return poserr, hidden1
        elif self.fwd_delay>0:
            return poserr[(self.fwd_delay):], hidden1[:(-self.fwd_delay)], x[:(-self.fwd_delay)]


def run_model(model,params,data,fb=True,dopert=0):
    if not fb:
        state_dict = model.state_dict()
        save = state_dict['feedback.weight'].detach().clone()
        state_dict['feedback.weight'] *= 0
        model.load_state_dict(state_dict)

    from dataset import perturb
    # SETUP ############
    testdata = data['test_set']
    tout = testdata['target'][:,:,2:]              # Target trajectories [n_trials, tstep, [x_velocity, y_velocity]]
    tstim = testdata['stimulus']                   # Input stimuli
    stim = torch.Tensor(tstim.transpose(1,0,2)).type(model.dtype)
    pert = torch.zeros(tout.transpose(1,0,2).shape).type(model.dtype)
    if dopert==1: # if perturbation type is random pushes
        pert = perturb(pert,stim.shape[1],params['p1'])
    # RUN ############
    testout,testl1 = model(stim,pert)         # Forward pass
    error = testout.cpu().detach().numpy().transpose(1,0,2)               # Position error
    testout = model.get_output(testl1) + pert                             # Adds perturbations back to output (produced velocities)
    # SAVE ############
    output = testout.cpu().detach().numpy().transpose(1,0,2)              # Produced velocities
    activity = testl1.cpu().detach().numpy().transpose(1,0,2)
    pert = pert.cpu().detach().numpy().transpose(1,0,2)
    state_dict = model.state_dict()
    alive_mask = state_dict['alive_mask']
    dic = {'target':tout,'stimulus':tstim,'error':error,
            'peak_speed':testdata['peak_speed'],'pert':pert,
            'activity':activity,'output':output, 'alive_mask':alive_mask}
    if 'tids' in testdata.keys():
        dic.update({'tid':testdata['tids']})
    
    if not fb:
        state_dict = model.state_dict()
        state_dict['feedback.weight'] = save
        model.load_state_dict(state_dict)
    return dic
    
def test(model,data,params,savname,lc,dopert,dataC):
    dic = run_model(model,params,data,dopert=dopert)
    np.save(savname+'testing',dic)
    
    # PLOTS ############
    import matplotlib.pyplot as plt
    import sklearn.metrics as met
    
    # setup
    cols = [plt.cm.magma(i) for i in np.linspace(0.1,0.9,8)]
    dt = params['model']['dt']
    
    # helper functions
    def get_pos(vel):
        pos = np.zeros(vel.shape)
        for j in range(vel.shape[1]):
            if j==0:
                pos[:,j] = dt*vel[:,j]
            else:
                pos[:,j] = pos[:,j-1] + dt*vel[:,j]
        return pos
           
    def plot_traj(datatemp,tid,xoffset=0):
        output = datatemp['output']
        target = datatemp['target']
        pos = get_pos(output)
        posT = get_pos(target)
        for j in range(pos.shape[0]):
            plt.plot(xoffset+pos[j,:,0],pos[j,:,1],color=cols[tid[j]],alpha=0.2)
        utid = np.unique(tid)
        for j in range(utid.size):
            idx = np.argwhere(utid[j]==tid)[0]
            plt.scatter(xoffset+posT[idx,-1,0],posT[idx,-1,1],facecolor='None',edgecolor='k',
                        marker='s',zorder=50)
        
    idx = 5
    result = dic
    plt.figure(figsize=(4,6),dpi=150)
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    
    plt.subplot(3,2,1)
    plt.title('Example Trial %d'%idx)
    plt.plot(result['stimulus'][idx])
    plt.ylabel('Stimulus')
    
    plt.subplot(3,2,2)
    plt.plot(lc)
    plt.ylabel('Loss')
    plt.xlabel('Training epochs')
    velT = result['target']
    vel = result['output']
    posT = get_pos(velT)
    pos = get_pos(vel)
    score = met.explained_variance_score(posT.reshape(-1,2),pos.reshape(-1,2))
    plt.title('VAF=%.2f'%score)
    
    plt.subplot(3,2,3)
    plt.plot(result['target'][idx],label='Target')
    plt.plot(result['output'][idx],'--',label='Produced')
    plt.ylabel('Output')
    
    plt.subplot(3,2,4)
    model_wFB = run_model(model,params,dataC)
    model_woFB = run_model(model,params,dataC,fb=False)
    tid = model_woFB['tid']
    
    plot_traj(model_wFB,tid)
    plt.text(0,10,'wFB',ha='center')
    
    plot_traj(model_woFB,tid,xoffset=30)
    plt.text(30,10,'woFB',ha='center')
    
    plt.xlim(-8,38)
    plt.ylim(-18,18)
    plt.axis('off')
        
    plt.subplot(3,2,5)
    plt.ylabel('Neurons')
    plt.imshow(result['activity'][idx,:,:].T,aspect='auto')
    plt.xlabel('Time bins')
    
    plt.subplot(3,2,6)
    plt.hist(result['activity'].ravel(),histtype='step',color='k')
    plt.xlabel('Neural activity')
    plt.ylabel('Histogram')
    
    plt.savefig(savname+'summary.svg',bbox_inches='tight')
    plt.savefig(savname+'summary.png',dpi=300,bbox_inches='tight')
    
