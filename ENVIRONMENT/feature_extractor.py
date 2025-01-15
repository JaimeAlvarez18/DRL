import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
class Layer(nn.Module):
    def __init__(
        self,
        input_dim : int,
        output_dim : int,
        num_heads : int
    ) -> None:
        """
        Basic Encoder module to process features.
        Parameters:
        -----------
            input_dim (int):
                Input embedding dim of features.
            output_dim (int):
                Output embedding dim of features.
            num_heads (int):
                Number of heads in Multihead-Attention.
        Returns:
        --------
            None
        """
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.num_heads=num_heads
        self.bool = False if input_dim != output_dim else True
        self.attention=nn.MultiheadAttention(embed_dim=input_dim,num_heads=self.num_heads)
        self.norm=nn.LayerNorm(input_dim)
        self.linear=nn.Linear(input_dim,output_dim)
        self.norm2=nn.LayerNorm(output_dim)
    def forward(
        self,
        input : torch.tensor
    ) -> torch.tensor :
        """
        Forward function of the module. It passes the input through all layers.
        Parameters:
        -----------
            input (torch.tensor):
                Input to be computed.
        Returns:
        --------
            out1 (torch.tensor):
                Output embeddings.
        """
        out1,_=self.attention(input,input,input)
        out1+=input
        out11=self.norm(out1)
        out1=self.linear(out11)
        if self.bool:
            out1+=out11
        out1=self.norm2(out1)
        return out1
    

class Encoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space : dict,
        total_cells : int,
        embedded_dim_big : int,
        n_encoders_concatenated : int,
        n_encoders_big : int,
        n_encoders_small : int,
        embedded_dim_small : int,
        features_dim : int,
        num_heads_small : int,
        num_heads_big : int,
        num_heads_concatenated : int
    ) -> None:
        """
        Full Features Extractor of the PPO algorithm. It stacks several Layers to process inputs.
        Parameters:
        -----------
            observation_space (dict):
                Current state of the observation space of the environment.
            total_cells (int):
                Total number of cells, including padding.
            embedded_dim_big (int):
                Embedding dimension for the grid cells.
            n_encoders_big (int):
                Number of Layers stacked to process raw grid cells.
            n_encoders_small (int):
                Number of Layers stacked to process both raw vertices of the layout and raw tractor position.
            embedded_dim_small (int):
                Embedding dimension for both vertices of the layout and tractor position.
            features_dim (int):
                Output embedding dimension.
            num_heads_small (int):
                Number of heads to use in Layers that process both raw vertices of the layout and raw tractor position.
            num_heads_big (int):
                Number of heads to use in Layers that process raw grid cells.
            num_heads_concatenated (int):
                Number of heads to use in Layers that process the concatenation of grid cells, layout vertices and tractor position embeddings.
        Returns:
        --------
            None 
        """
        super().__init__(observation_space, features_dim=features_dim)
        self.input_dim=embedded_dim_small
        self.embedded_dim_big=embedded_dim_big
        self.n_encoders_big=n_encoders_big
        self.n_encoders_small=n_encoders_small
        self.output_dim=features_dim
        self.n_layers=n_encoders_concatenated
        self.num_heads=num_heads_small
        self.num_heads_big=num_heads_big
        self.num_heads_concatenated=num_heads_concatenated
        self.models_total = nn.ModuleList([Layer(self.output_dim,self.output_dim,self.num_heads_concatenated) for _ in range(self.n_layers)])

        self.layer4x2=nn.ModuleList([Layer(self.input_dim,self.input_dim,self.num_heads) for _ in range(self.n_encoders_small)])

        self.layer5x2=nn.ModuleList([Layer(self.input_dim,self.input_dim,self.num_heads) for _ in range(self.n_encoders_small)])
        
        self.layer310x3=nn.ModuleList([Layer(self.embedded_dim_big,self.embedded_dim_big,self.num_heads_big) for _ in range(self.n_encoders_big)])

        self.extractor_5x2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 2, self.input_dim))
        
        self.extractor_4x2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 2, self.input_dim))
        
        self.extractor_310x3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_cells * 3, self.embedded_dim_big))
        
    def forward(
        self,
        x : torch.tensor
    ) -> torch.tensor:
        """
        Forward function of the module. It passes the input through all layers.
        Parameters:
        -----------
            x (torch.tensor):
                Input to be computed.
        Returns:
        --------
            concatenated (torch.tensor):
                Output embeddings.
        """

        features_4x2 = self.extractor_4x2(x["current_position"])
        features_5x2 = self.extractor_5x2(x["edges"])
        features_310x3 = self.extractor_310x3(x["grid"])

        for model in self.layer4x2:
            features_4x2 = model(features_4x2)

        for model in self.layer5x2:
            features_5x2 = model(features_5x2)

        for model in self.layer310x3:
            features_310x3 = model(features_310x3)

        concatenated = torch.cat([features_4x2, features_5x2, features_310x3], dim=-1)

        for model in self.models_total:
            concatenated = model(concatenated)

        return concatenated