import torch
from torch.distributions import Normal
import torch.nn as nn
import numpy as np

from math import pi
from scipy.special import logsumexp

class cluster_GMM_Dist():
    def __init__(self, clustering_model, vars, method = "kmeans", manual_weights=None, normalize_direction=False) -> None:
        if method == "kmeans":
            self.clustering_model = clustering_model
            self.num_clusters = clustering_model.n_clusters
            self.means = torch.Tensor(clustering_model.cluster_centers_.reshape(self.num_clusters, 12, 2)).cuda()
            self.manual_weights = manual_weights
            self.normalize_direction = normalize_direction
            
            self.get_sampleNum(20)

            self.vars = vars
        else:
            raise NotImplementedError

    def get_sampleNum(self, n_samples):
        if self.manual_weights is None:
            _, label_counts = np.unique(self.clustering_model.labels_, return_counts=True)
            weights = label_counts / np.sum(label_counts)
            self.sample_nums = np.round(n_samples * weights, 0).astype(int)

            while(np.sum(self.sample_nums) != n_samples):
                decimal = 20*weights - self.sample_nums
                range_index = np.cumsum(self.sample_nums)
                if np.sum(self.sample_nums) > n_samples:
                    index = np.argmin(decimal)
                    index_group = np.where(index<range_index)[0][0]
                    self.sample_nums[index_group] -= 1
                else:
                    index = np.argmax(decimal)
                    index_group = np.where(index<range_index)[0][0]
                    self.sample_nums[index_group] += 1
        else:
            weights = np.array(self.manual_weights)
            self.sample_nums = weights * int(n_samples / weights.sum())
        
    def set_dist(self, base_pos):
        batch_size = base_pos.shape[0]
        self.dist = []
        if not self.normalize_direction:
            direction = base_pos
            angles = torch.arctan2(direction[:,1], direction[:,0])
            rotate_matrix = torch.stack([torch.cos(angles), torch.sin(angles),
                                        -torch.sin(angles), torch.cos(angles)]).reshape(2,2,-1).permute([2,0,1])    # (B,2,2)
            project_matrix = torch.abs(rotate_matrix)

            means_rotate = torch.matmul(self.means.unsqueeze(1), rotate_matrix.unsqueeze(0))
            new_means_rotate = torch.cat((direction.unsqueeze(0).unsqueeze(-2).expand(self.num_clusters,-1,-1,-1), means_rotate[:,:,:-1,:]), dim=-2)
            # new_means_rotate = means_rotate     # allfut
            vars_rotate = torch.matmul(self.vars.unsqueeze(1), project_matrix.unsqueeze(0))

            for i in range(self.num_clusters):
                self.dist_i = Normal(new_means_rotate[i], torch.clamp(vars_rotate[i], min=1e-4))
                self.dist.append(self.dist_i)
        else:
            for i in range(self.num_clusters):
                self.dist_i = Normal(self.means[i].unsqueeze(0).repeat(batch_size, 1, 1), torch.clamp(self.vars[i].unsqueeze(0).repeat(batch_size, 1, 1), min=1e-4))
                self.dist.append(self.dist_i)

        # print(new_means_rotate.shape)     # (8,1024,12,2)
        # print(vars_rotate.shape)          # (8,1024,2)
        # print(self.vars.shape)        # (8,12,2)
        
        # construct dist
        
        
        
    def sample(self, n_sample=20):
        if np.sum(self.sample_nums) != n_sample:
            self.get_sampleNum(n_sample)
        samples = []
        for i,d in enumerate(self.dist):
            samples_i = d.sample((n_sample,))  # (20,B,12,2)
            samples.append(samples_i[:self.sample_nums[i]])    

        samples = torch.cat(samples, dim=0).permute(1,0,2,3)    # (B,20,12,2)
        return samples
    
    def sample_mean(self, sample_num=20):
        samples = []
        for i,d in enumerate(self.dist):
            samples_i = d.loc.unsqueeze(0).expand([sample_num,-1,-1,-1])
            samples.append(samples_i[:self.sample_nums[i], :, 0, :])    # first prediction

        samples = torch.cat(samples, dim=0).permute(1,0,2)
        return samples
    
    def log_prob(self, base_pos, x, u):
        if not self.normalize_direction:
            ## rotate
            direction = base_pos
            angles = -torch.arctan2(direction[:,1], direction[:,0])
            rotate_matrix = torch.stack([torch.cos(angles), torch.sin(angles),
                                        -torch.sin(angles), torch.cos(angles)]).reshape(2,2,-1).permute([2,0,1])
            x_rotate = torch.matmul(x, rotate_matrix)

            clusters = self.clustering_model.predict(x_rotate.detach().reshape(-1,24).cpu().numpy())
        else:
            clusters = self.clustering_model.predict(x.detach().reshape(-1,24).cpu().numpy())
        ## cluster
        
        

        # probs = []
        # for d in self.dist:
        #     prob_i = d.log_prob(x)
        #     probs.append(prob_i.unsqueeze(0))
        # probs = torch.cat(probs)
        # traj_log_probs = probs.sum(-1).sum(-1)
        # _, max_traj = traj_log_probs.max(0)

        # global gmm_fit_global, count
        # gmm_fit = np.array([max_traj.cpu().numpy(), clusters])
        # gmm_fit_global.append(gmm_fit)
        # count += 1
        # if count == 10:
        #       print("10")
            
        ## compute
        log_prob_ = torch.empty_like(u)
        for i in range(self.num_clusters):
            dist = self.dist[i]
            mask = clusters == i
            log_prob_[mask] = dist.log_prob(u)[mask]
        
        return log_prob_