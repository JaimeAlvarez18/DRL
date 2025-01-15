import matplotlib
matplotlib.use('TkAgg')
from environment import Environment
from shapely.geometry import Polygon as P
from stable_baselines3 import PPO
import yaml


from masked_PPO_policy import MaskedEnvWrapper, MaskedPolicy

from feature_extractor import Encoder
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Example usage:
if __name__ == "__main__":
    file = open("config.yaml", "r")
    config = yaml.safe_load(file)
    
    # Load all configuration
    policy=config['Policy']['Masked']

    timesteps = config['Training']['Time_step_per_iter']
    n_iters = config['Training']['N_iters']
    name_to_save=config['Training']['name_saved_model']
    device=config['Training']['device']
    new=config['Training']['new']
    name_to_load=config['Training']['load_model']


    vf=config['Action_predictor']['net_arch_vf']
    pi=config['Action_predictor']['net_arch_pi']

    clip_range=config['PPO_args']['clip_range']
    gae_lambda = config['PPO_args']['gae_lambda']
    ent_coef=config['PPO_args']['ent_coef']
    learning_rate=config['PPO_args']['learning_rate']
    

    embedded_dim_big=config['Feature_extractor']['embedded_dim_big']
    n_encoders_concatenated=config['Feature_extractor']['n_encoders_concatenated']
    n_encoders_big=config['Feature_extractor']['n_encoders_big']
    n_encoders_small=config['Feature_extractor']['n_encoders_small']
    embedded_dim_small=config['Feature_extractor']['embedded_dim_small']
    num_heads_concatenated=config['Feature_extractor']['num_heads_concatenated']
    num_heads_big=config['Feature_extractor']['num_heads_big']
    num_heads_small=config['Feature_extractor']['num_heads_small']
    features_dim= embedded_dim_small*2+embedded_dim_big

    print('Creating env...')

    # Create environment
    env=Environment(config)
    total_cells=env.grid.total_cells_padded
    file.close()
    
    print('Env has been reset as part of launch')

    # Configuration for feature extractor
    policy_kwargs = dict(
        features_extractor_class=Encoder,
        features_extractor_kwargs={
            'total_cells' : total_cells,
            'embedded_dim_big' : embedded_dim_big,
            'n_encoders_concatenated':n_encoders_concatenated,
            'n_encoders_big':n_encoders_big,
            'n_encoders_small':n_encoders_small,
            'embedded_dim_small':embedded_dim_small,
            'features_dim':features_dim,
            'num_heads_small':num_heads_small,
            'num_heads_big':num_heads_big,
            'num_heads_concatenated': num_heads_concatenated
        },
        net_arch=dict(pi=pi,vf=vf)
    )

    # If new training
    if new:

        # If using masked policy
        if policy:

            env = make_vec_env(lambda: MaskedEnvWrapper(env), n_envs=1)
            model = PPO(MaskedPolicy, env, policy_kwargs=policy_kwargs, verbose=1,device=device,clip_range=clip_range,gae_lambda=gae_lambda,ent_coef=ent_coef,learning_rate=learning_rate)
        
        # If not using masked policy
        else:
            env = DummyVecEnv([lambda: env])
            model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1,device=device,clip_range=clip_range,gae_lambda=gae_lambda,ent_coef=ent_coef,learning_rate=learning_rate)
    
    # If re-training
    else:

        if policy:
            env = make_vec_env(lambda: MaskedEnvWrapper(env), n_envs=1)
            
        else:
            env = DummyVecEnv([lambda: env])
        model = PPO.load(name_to_load,env,device=device)

    #Training lopp.
    for iter in range(n_iters):
        print('Iteration ', iter,' is to commence...')
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False )
        print('Iteration ', iter,' has been trained')
        model.save(f"{name_to_save}")
        print(f'Trained model saved in {name_to_save}')

    


    
    




        