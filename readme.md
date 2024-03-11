#  Differentially Private Federated Learning with Low-Rank Adaptation (LoRA)

This repository contains the code for our [*winning entry*](https://benchmarks.elsa-ai.eu/?ch=2&com=evaluation&task=2) in PFL-DocVQA Competition 2023 ([Track 2](https://benchmarks.elsa-ai.eu/?ch=2&com=introduction)). Our contribution is very simple. We applied LoRA to the provided baseline. By reducing the no. of training parameters, our method significantly reduces both total communication costs and the overall noise added to the model during training. Below, we provide a simple and straightforward  explanation of our method. 

**Authors**: Ragul N<sup>[1]#</sup>, Sivasanjai GA<sup>[1]</sup>, Rintu Kutum<sup>[1][2]*</sup>

**Affiliations**:

<sup>[1]</sup>Department of Computer Science, Ashoka University

<sup>[2]</sup>Trivedi School of Biosciences, Ashoka University

<sup>\#</sup> First author

<sup>\*</sup> Corresponding author. rintu.kutum@ashoka.edu.in

## Priavte Machine Learning with Diffrential Privacy:
### Differential Privacy: Intuition and Formula
<div style="text-align: justify;">
At its core, differential privacy(DP) provides a mathematical guarantee that the presence or absence of a single data point does not substantially alter the output distribution of a computation. The intuition lies in introducing randomness to mask individual contributions, making it challenging for an observer to discern any specific influence. The formal definition is expressed through the privacy parameter, denoted as ε (epsilon), which quantifies the maximum allowable change in output probabilities caused by the inclusion or exclusion of a single data point.

Mathematically, a randomized algorithm $M$ is $(\epsilon, \delta)$-differentially private if, for all neighboring datasets \(D\) and \(D'\) (differing in a single data point), and for all possible outcomes \(S\):

Mathematically, a randomized algorithm A is ε-differentially private if, for all neighboring datasets D and D' (differing in a single data point), and for all possible outcomes $\Theta$:

$$ Pr[M(D) \in \Theta] \leq e^\varepsilon \cdot Pr[M(D') \in \Theta] + \delta $$

where $\epsilon$ controls the privacy level and $\delta$ accounts for a small failure probability.

</div>


<div style="text-align: justify;">

### Gaussian Mechanism: 
The Gaussian Mechanism is method for achiving diffrenital privacy.  Given a function f, the Gaussian Mechanism adds noise sampled from a Gaussian distribution with standard diviation propotional to the sensitivity of f. The sensitivity measures how much the output of the function can change due to the addition or removal of a single data point.

Mathematically, for a function f with sensitivity Δf, the Gaussian Mechanism perturbs the function as follows:

$$ f(D) + \text{Noise} \sim \mathcal{N}(0, \sigma^2) $$

where $\sigma \propto \frac{\Delta f}{\epsilon}$ determines the scale of the added noise. 

</div>

<div style="text-align: justify;">

### Stochastic Gradient Descent: 

Stochastic Gradient Descent (SGD) is a widely-used optimization algorithm for training machine learning models. The basic idea is to iteratively update the model parameters by moving in the direction of the negative gradient of the loss function. The update rule for a parameter $\theta$ in the $t$-th iteration is given by:

$$ \theta_{t+1} = \theta_t - \eta \nabla f(\theta_t; D_t) $$

where $\eta$ is the learning rate and $\nabla f(\theta_t; D_t)$ is the gradient of the loss function with respect to the model parameters computed on a subset $D_t$ of the training data.

### Differentially Private Gradient Descent Algorithm:

To make SGD differentially private, we introduce the Gaussian Mechanism to the gradient updates. The differentially private gradient update for parameter $\theta$ in the $t$-th iteration becomes:


$$ \theta_{t+1} = \theta_t - \eta \left(\text{Clip}\left(\nabla f(\theta_t; D_t),S\right) + \mathcal{N}(0, \sigma I)\right) $$

Here,  the $\text{Clip}(\cdot, S)$ operation ensures that gradients are scaled down if their magnitude exceeds a predefined sensitivity S, preventing overly large updates that could compromise privacy. And $\sigma \propto \frac{S}{\epsilon}$, determines the scale of the added noise.

## Federated Learning with Local Diffrential Privacy: 

In federated learning, the objective is to train a global model across decentralized clients while preserving the privacy of individual data on each client. To achieve this, the combination of FedAvg and local differential privacy can be employed. 

### Local Differential Privacy at Each Client
Local differential privacy focuses on injecting noise at the individual client level, providing privacy protection for local datasets. Each client independently applies differential privacy mechanisms to its local data before communicating with the central server. Mathematically, the local differential privacy mechanism for a client $i$ can be represented as:

$$ \text{Local Mechanism at Client } i: \quad f(D_i) + \mathcal{N}(0, \sigma_i^2 I) $$

Here, $f(D_i)$ represents the local update at client $i$, and $N(0, \sigma_i^2 I)$ denotes Gaussian noise added for differential privacy, where $\sigma_i$ determines the magnitude of noise for client $i$.

### FedAvg:

**Federated Averaging (FedAvg)**

FedAvg is a federated learning algorithm that involves iterative model updates and aggregation across multiple clients. The basic steps of FedAvg are as follows:

1. **Initialization**: Initialize a global model on the central server.

2. **Client Update**: Randomly select a subset of clients, each denoted by $C_i$, where $i$ ranges over the selected clients. Each client in the subset performs a local update using its private data, generating a model update denoted by $\Delta \theta_i$.

3. **Model Aggregation**: The central server aggregates the model updates from the selected clients using a weighted average:

   $$\Delta \theta = \frac{1}{|C|} \sum_{i \in C} w_i \Delta \theta_i $$

   Here, $w_i$ represents the weight assigned to each client in the subset, and $|C|$ is the size of the subset.

4. **Global Model Update**: The central server updates the global model using the aggregated update:

   $$\theta_{\text{global}} = \theta_{\text{global}} - \eta \Delta \theta $$

   where $\eta$ is the learning rate.

### Incorporating Local DP-SGD into FedAvg
To introduce local differential privacy in FedAvg, we add noise at the client level during the model update step. Specifically, each client perturbs its local update using local DP-SGD before sending the model update to the central server.

The local DP-SGD update at client $i$ in the $t$-th iteration can be expressed as follows:

$$ \Delta \theta_i^{\text{local}} = \Delta \theta_i - \eta \cdot \text{Clip}\left(\nabla f(\theta_i; D_i), S \right) + \mathcal{N}(0, \sigma I) $$

Here, $\Delta \theta_i$ is the local update without differential privacy, $\nabla f(\theta_i; D_i)$ is the gradient of the loss function with respect to the model parameters computed on the local dataset $D_i$, $\text{Clip}(\cdot, S)$ denotes the gradient clipping operation, and $\mathcal{N}(0, \sigma I)$ represents the Gaussian noise added for local differential privacy, where $\sigma \propto \frac{S}{\varepsilon}$.

Now, during the federated averaging step, the central server aggregates the locally perturbed updates from the selected clients using a weighted average:

$$ \Delta \theta = \frac{1}{|C|} \sum_{i \in C} w_i \Delta \theta_i^{\text{local}} $$

The global model is then updated with the aggregated perturbed update:

$$ \theta_{\text{global}} = \theta_{\text{global}} - \eta \Delta \theta $$

This modification ensures that each client contributes to the global model in a privacy-preserving manner, incorporating local differential privacy through noise addition at the client level. The weights \(w_i\) can be adjusted to reflect the contribution of each client, considering factors such as data size or computation capabilities.

## Problems in DP Federated Learning: 

### Communication Cost with FedAvg:
In Federated Learning, the communication cost can be a significant challenge. Clients communicate model updates to a central server during each round, leading to potential bandwidth issues, especially when dealing with numerous or resource-constrained clients.

### Privacy-Utility Tradeoff: 
The introduction of noise in DP machine learning to ensure privacy comes with a trade-off. The added noise, crucial for differential privacy, leads to performance degradation in the model. Balancing privacy and utility becomes a challenge.

## LoRA with DP-FedAvg:
### What is LoRA?
Low Rank Adaptation (LoRA) is a parameter efficient fine-tuning technique that significantly reduces the number of trainable parameters needed for fine-tuning. For each pre-trained weight matrix $W_0 \in R^{d \times k}$, LoRA constrains its update by representing it with a low-rank decomposition: $W_0 + \Delta W = W_0 + BA$. Here, $B \in R^{d \times r}$ and $A \in R^{r \times k}$, with $r \leq \min(d, k)$ being the rank parameter. During training, $W_0$ remains frozen and does not receive gradient updates, while $A$ and $B$ contain trainable parameters.

### Why LoRA for DP-FedAvg?

1. **Reduced Communication Cost**:
LoRA reduces the number of parameters involved in model updates, leading to a decrease in the communication cost between clients and the central server in the federated learning setting. 

2. **Reduced Noise Addition:** Differential privacy only requires noise addition to the model update parameters. By reducing size of the model update, LoRA also reduces the total noise added to the model. 

### Applying LoRA to Baseline:
We applied LoRA reparameterization only to the Transformer attention weights $W_{q,v}$ and freeze all other weights. For rank r, we experimented with diffrent value and settled on r=16. This reduces model update size from 1.12GB to ~200MB. 

After Applying LoRA, we used the following hyperparameters for training.

|     |  $\epsilon =1$  | $\epsilon =4$   | $\epsilon =8$  
| ------:| :-----: | :---: | :----:| 
| Noise Multiplier |  1.21  | 0.695 | 0.553
|  Sensitivity ($s$) |   0.5   | 0.5 | 0.5 
|  Clients per round ($K$)   | 2 |   2 | 2
|  Providers per client($M$)  | 45|  45 | 45
|  Total Rounds($N$)   | 30|   30 | 30
|  Delta($\delta$)   | $1 \times 10^{-5}$|   $1 \times 10^{-5}$ | $1 \times 10^{-5}$

### Results
The application of LoRA yielded significant performance improvements across all three privacy budgets. The table below compares the accuracy achieved using LoRA against a baseline model:

| Privacy Budget    |  LoRA  | Baseline 
| :------: | :-----: | :---: |
| $\epsilon =1$  |  57.4%   |  46.2%
|  $\epsilon =4$  |   59.75%   | 48.3%
|  $\epsilon =8$     | 60.4% |   50.3%


</div>

