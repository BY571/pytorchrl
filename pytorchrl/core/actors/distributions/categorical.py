import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ..neural_networks.feature_extractors.utils import init


class Categorical(nn.Module):
    """
    Categorical probability distribution.

    Parameters
    ----------
    num_inputs : int
        Size of input feature maps.
    num_outputs : int
        Number of options in output space.

    Attributes
    ----------
    linear: nn.Module
        Maps the incoming feature maps to probabilities over the output space.
    """
    def __init__(self, num_inputs, num_outputs, multi_discrete=False):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)
        if not multi_discrete:
            self.linear = init_(nn.Linear(num_inputs, num_outputs))
            self.multi_discrete = multi_discrete
        else:
            # creates output heads depending on the number of action dimensions with the output features as the number of sub action dimensions (3)
            self.linear = [init_(nn.Linear(num_inputs, 3)) for head in range(num_outputs[0])]
            self.num_outputs = num_outputs
            self.multi_discrete = multi_discrete

    def forward(self, x, deterministic=False):
        """
        Predict distribution parameters from x (obs features) and return
        predictions (sampled and clipped), sampled log
        probability and distribution entropy.

        Parameters
        ----------
        x : torch.tensor
            Feature maps extracted from environment observations.
        deterministic : bool
            Whether to randomly sample from predicted distribution or take the mode.

        Returns
        -------
        pred: torch.tensor
            Predicted value.
        clipped_pred: torch.tensor
            Predicted value (clipped to be within [-1, 1] range).
        logp : torch.tensor
            Log probability of `pred` according to the predicted distribution.
        entropy_dist : torch.tensor
            Entropy of the predicted distribution.
        """

        # Predict distribution parameters
        if not self.multi_discrete:
            x = self.linear(x)
        else: 
            x = torch.cat([head(x) for head in self.linear])
            #x = x.reshape((self.num_outputs[0], 3))

        # Create distribution and sample
        dist = torch.distributions.Categorical(logits=x)
        self.dist = dist # ugly hack to handle sac discrete case

        if deterministic:
            pred = clipped_pred = dist.probs.argmax(dim=-1, keepdim=True)
        else:
            pred = clipped_pred = dist.sample().unsqueeze(-1)

        # Action log probability
        # logp = dist.log_prob(pred.squeeze( -1)).unsqueeze(-1)
        logp = dist.log_prob(pred.squeeze(-1)).view(pred.size(0), -1).sum(-1).unsqueeze(-1)
        if self.multi_discrete:
            pred = pred.squeeze().unsqueeze(0)
            logp = logp.squeeze().unsqueeze(0)
        # Distribution entropy
        entropy_dist = dist.entropy().mean()
        return pred, clipped_pred, logp, entropy_dist


    def evaluate_pred(self, x, pred):
        """
        Return log prob of `pred` under the distribution generated from
        x (obs features). Also return entropy of the generated distribution.

        Parameters
        ----------
        x : torch.tensor
            obs feature map obtained from a policy_net.
        pred : torch.tensor
            Prediction to evaluate.

        Returns
        -------
        logp : torch.tensor
            Log probability of `pred` according to the predicted distribution.
        entropy_dist : torch.tensor
            Entropy of the predicted distribution.
        """

        # Predict distribution parameters
        if not self.multi_discrete:
            x = self.linear(x)
        else: 
            x = torch.cat([head(x) for head in self.linear])
            x = x.reshape((pred.shape[0], self.num_outputs[0], 3)) 

        # Create distribution
        dist = torch.distributions.Categorical(logits=x)

        # Evaluate log prob of under dist
        if not self.multi_discrete:
            logp = dist.log_prob(pred.squeeze(-1)).unsqueeze(-1).sum(-1, keepdim=True)
        else:
            logp = dist.log_prob(pred.squeeze(-1)).unsqueeze(-1).sum(-1, keepdim=True).squeeze()

        # Distribution entropy
        entropy_dist = dist.entropy().mean()

        return logp, entropy_dist