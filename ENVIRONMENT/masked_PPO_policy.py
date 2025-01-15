from stable_baselines3.common.distributions import CategoricalDistribution, Distribution

from stable_baselines3.common.policies import ActorCriticPolicy
from utils import *
import torch
import gymnasium as gym
import yaml
class MaskedEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MaskedEnvWrapper, self).__init__(env)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def get_action_mask(self):
        return self.env.get_action_mask()


class MaskedPolicy(ActorCriticPolicy):
    def __init__(self,*args, **kwargs):
        super(MaskedPolicy, self).__init__(*args, **kwargs)
        file = open("config.yaml", "r")
        config = yaml.safe_load(file)
        self.hop = config['Layout']['hop']
        self.rotation_angle = config['Layout']['rotation_angle']
        self.number_actions = config['Policy']['Number_of_actions']
        file.close()

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)


        vertices=obs['edges'].to('cpu').numpy()
        vertices=vertices.squeeze(0)
        rectangle = obs['current_position'].to('cpu').numpy()
        rectangle=rectangle.squeeze(0)
        self.mask=self.get_action_mask(rectangle,vertices)

        
        
        values = self.value_net(latent_vf)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)

        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        return actions, values, log_prob
    
    def _get_action_dist_from_latent(self, latent_pi) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """

        mean_actions = self.action_net(latent_pi)

        self.mask = self.mask.clone().detach().to(device='cuda')

        mean_actions=mean_actions-self.mask*float(9999999999999999999999999999999999)

    
        if isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")


    def get_action_mask(self,rectangle,vertices):
        
        
        # Example: Mask out action 2 and 4
        l=[]
        for i in range(2):
            for j in [self.rotation_angle,-self.rotation_angle]:
                rectangle1=rotate_around_vertex(rectangle,j,i)
                out=check_out_edge(rectangle1,vertices)
                if out:
                    l.append(1)
                else:
                    l.append(0)
        for i in [self.hop,-self.hop]:
        
            direction=get_forward_direction(rectangle)
            rectangle1=move_rectangle(rectangle,direction,i)
            out=check_out_edge(rectangle1,vertices)
            if out:
                l.append(1)
            else:
                l.append(0)
        if self.number_actions == 10:
            for i in [2,3]:
                for j in [self.rotation_angle,-self.rotation_angle]:
                    rectangle1=rotate_around_vertex(rectangle,j,i)
                    out=check_out_edge(rectangle1,vertices)
                    if out:
                        l.append(1)
                    else:
                        l.append(0)
        return torch.tensor(l, dtype=torch.float32)