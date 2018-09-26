delayed reward


knowledge is the model, process inputs to outputs

## frameword
###environment vs agent
state (observation) - action - reward
time 0: s_0, a_0
time 1: r_1, s_1, a_1

### goal for agent
maximize expected cumulative reward
the designed reward is called __the reward hypothesis__
for a deepmind robot, reward for its every step: +moved distance, +speed, --joint force, -moving aside, -distance off the center

### Return
return (G_t): cumulative rewards from time t onward
action_t = argmax(expected(G_t))
discount rate, discount the future returns with gamma 

### episodic vs. continuning tasks
task is a RL problem
episodic: have well-defined start and end
the episode end at terminal state
sparse reward: uninformative reward 


## Question Frame: Markov Decision Process
example of recycle robot
* actions A = [search cans, recharge at dock, wait]
* state S = [battery high, low], all state without terminal state
* transitions of actions that connect between states
* reward: search=4, wait=1, rescue=-3 <- no need for "rescue" state, express it via rewards
no previous actions is recorded, nor do they affects, called __one-step dynamics__
* math expression as P = [possibility p(new state s', reward r | given state s, action a)]
MDP is defined by A, S, P and gamma

## Solution Framework
find optimal policy
* Model-based Solution
MDP model is known
__policy / value iteration__
* Model-free Solution
MDP unknown, need exploration
__Monte Carlo__, __Temporal Difference__


### policy: 
deterministic: map from state to action
stochastic: same map with possibility
* comparing policies:
if v_p1(s) >= v_p2(s) for all s in S, p1 is better than p2
optimal p if p is better than all p, and v_p is denoted as v*

### state value function v
for each state, have a corresponding expected return number if agent start at this state, and follow policy all the time
value v_p(s) = E_p(G | s)
recurrsive v_i = v_i+1 + r_i+1, v_terminal = 0
* __Bellman Expectation Eq.__
v_p(s) = E_p(R_t+1 + gamma * v_p(s+1) | s)

### action value function __Q__
add action to state value func
value q_p(s, a) = E_p(G, | s, a)
for given p: v_p(s) = q_p(s, a=p(s))

## optimization
optimal p if p is better than all p, and v_p is denoted as v*
for given p* : v* (s) = q* (s, a=p(s))
so q* ==> p* by choosing a so that q* (s) = argmax_a(q(s,a))


### policy iteration
_assume MDP is clearly known, no need for exploitation, so-called dynamic programming setting_
_typically not the case, much easier than learning the whole MDP from interactions_
first, assume equal probable random policy, p(a) have same possibility
#### iterative policy evaluation
given policy, find value func.
make assumption to unknown state G, than iteratively loop over all states
stop criteria: break when change in G <= preset theta
#### policy improvement
given value func, find better policy
first calculate all q(s,a) from v(s)
update policy by the max action value argmax_a(q(s,a)) ___greedy policy___
break when policy stop changing or difference smaller than a threashold
a feedback loop is setup by again policy eval. the new policy to again update value func., 
___BELLMAN ALGORITHM ON GRAPH___

### Value iteration 
for each state in V:
	update V[s] to max of q <- because max of q w
	each sweep over the state space simultaneously performs policy evaluation and policy improvement
v_prev = v
v = policy_evaluation(policy_improvement(v))
break if v - v_prev reach threashold


## Monte Carlo RL algo
for episodic task
need to calculate the return to update return of Q(state,actions)
### Policy Evaluation /MC Prediction
given policy for to generate s and a, calculate Q(s,a)
average_reduce action value Q(s,a)= sum(g(s,a)) / totalN of (s,a) pairs in sampled episodes
* first-time vs. every-time: only count in first (s,a) in one episode, or take all (s,a) in the episode
* On-policy vs. off-policy: the agent interact with the environment by the same policy vs different policy that it seeks to evaluate (or improve)

### Policy Improvement / MC Control
greedy policy from DP, argmax_a will not work, because we need unbiased samples for policy evaluation, to explore MDP
* __Epsilon Greedy Policy__
eps = possibility to explore
* Exploration/Exploitation Trade-off
exploration: explore unknown hypothesis on action reacts
exploitation: exploit known info to choose optimal actions
favor explore at beginning, and exploit later
in practice: ___GLIE___: Greedy in the Limit with Infinite Exploration

### Action Value + Policy Iteration / MC Control
* running mean: update Q for each episode
q = s/n -> s = q * n
q_new =  (s + g)/(n+1) = q + 1/(n+1)(g - q)
as n increase, 1/(n+1) gets smaller and smaller, since initial values takes weights forever
* forgetful mean
set it to constant alpha to address most recent rewards, and gradually forget initials
q_new = (1-alpha) * q + alpha * g
in practice: ___constant-alpha GLIE MC Control___

## Temporal Difference RL
work for both coutinuing and episodic task
estimate possible return for every move
### TD Prediction: TD(0)
_BELLMAN + alpha_mean_
q_t = r + gamma * q_t+1 = g
Q_tnew = (1-alpha) * Q_t + alpha * g
Q_tnew = (1-alpha) * Q_t + alpha * (r + gamma * Q_t+1)
q == v == g for definition, but some Q is average-reduced q from Q
new estimate change from g to TD-target: r + gamma * Q_s+1, by bellman
no return so no need for episode, only need next state
q is updated after every step, so called ___One-step TD, TD(O)___
practically converge faster than MC prediction

### TD Control: SARSA(0)
in order to have Q_t+1, we need both s_t+1 and a_t+1 to get Q(s,a), so we need s_t, a_t, r_t+1, s_t+1, a_t+1 for each update
so called ___SARSA(0)___
both a_t and a_t+1 is determined as epsilon greedy method
* _Expected SARSA_
instead of epsilon-greedy to pick one, the method calculate the expected v from possibilities of different a_t+1
better performance
* both on-policy method
do as learn, good exploitation, less exploration, result Q is affected by epsilon value, better online-performance
_Sarsa achieves better online performance, but learns a sub-optimal "safe" policy._
can flexibly build explortory policy to act, and learn the optimal policy
can learn from demonstrations, or off-line

### TD Control: ___Q-Learning___ / SARSAMAX(0)
almost identical to SARSA, except a_t+1 is selected via pure greedy method, so q(s_t+1, a_t+1) = v(s_t+1), where v = argmax_a(q(s,a))
* off-policy method
good exploration, less exploitation, result Q is not affected by epsilon, more accurate learning
_Q-learning achieves worse online performance (where the agent collects less reward on average in each episode), but learns the optimal policy, and_


## Deep MDP
use nn to solve RL
### Infinite and Countinuous MDP
usually RL's MDP is finite and discrete on s, a, r
it allows dict/table data structure to do DP greedy algorithm
also allows iterations over states, actions in all RL algorithm
* continuous
most physical world examples: robot motions, have countinuous states and actions
deal with discretization and function approximation

### Discretization
some continuous environments can be discretized with bearable loss, and results can apply RL with little modification
draw grids on given space, and encode grid that contains given point as "on"; one grid is one feature and each sample has n_of_grids features
non-uniform D: adjust width of grid based on conditions
* Tile Coding: multiply layers of grids (tiles) to encode a continuous space, where each grid is one feature, each layer has its own set of "on" features
* Coarse Coding: instead of grid, drop random circles on the space
limitation: less accurate, may need huge amount of new features

### Function Approx.
directly estimate v based on s and a via approx func v' so that v'(state, Weights) = v(state) _the DEEP part_
to find v' and W, we first map state s to one-dimension vector feature X = x(s), and W share X's shape. v' = X dot W = v.
x() is called kernal functions, it can be linear or non-linear like x(s) = s^2
or like nn, we add a activation function over v', so that v = f(v'(s, w)) = f(v'(X dot W))
Use gradient descent to optimize W
to find q', we define 1-d vector X = x(s, a) and follow the same procedure
to find argmax_a(q'), turn 1-d W to 2-d [W] * action_number, and do [X(s,a1), (s,a2),...(s,an)] matmul [W] * n. if a is discrete, max(result), if a is continuous, then dunno yet

## DEEP RL Algos
while NN as supervised learning, deep RL algo also need labels to learn
* DEEP Model-Based Algo
Q'(s,a,W) is train to Q(s,a), as MDP is well understood
* DEEP Model-Free Algo
MC: use return G to update Q, so we train G'(s,a,w) to predict G, where G can be calculated, steps cannot be continuous
TD: use TD-target (r+gamma * s+1) to update Q with bellman, so we train TD-t'(s,a,w) to predict TD-t
on essence, traditional RL remember all q(s,a) via G or TD-t and get q by average, at every policy eval-improve iteration; deep RL train q'(s,a) against G or TD-t, at every iteration
the nature of NN approach can capture underlaying connections between states, like physical location coordinates, and fill in holes that (s,a) never appears

### Replay Memory Sampling
to prevent action-state correlation
imagine playing tennis, choose forehand/backhand for forehand side or backhand side. 
forehand reinforce straight ball to bounce back the forehand side ball for next timestep (high correlation between action and state, not rare), so loop goes one and backhand side ball becomes very rare
with nn nature, preference of forehand may be leared across all space, both sides
* solution: turn rl to sl
1. explore more, save [s, a, r, s+1, a+1] in a "replay memory" database
2. train Q'(sarsa) with random sampling from replay memory, so that forehand will not reinforce forehand side ball
3. apply any sl techniques when suitable

### DEEP Q LEARNING
init replay memory D with capacity N
init Q' with weights W
sampling: exploring SARSA and save to D
learning: train Q' with D in RL frame

* Fixed Weights
fix weights while updating weights, TODO not really get it
* Double Q Learning
prevent overestimate Q for argmax_a(s,a,w) when w is not stable yet
use two set of w as safety factor
* Priorized Replay Memory
prevent forgetting the rare and important SARSA
the metrics of importance could be the TD-Error, the larger it is the more we learn from that specific SARSA, so pri = TD-E
to prevent neglecting TD-Error = 0, where q have readily stablized, we can add a constant e, so pri = pri + e
to prevent overfitting we add a to possibility equation, where a=0 is uniform and a=1 is pri
hence when sampling the possibility p = pri^a/sum(pri^a)
* Dueling Network
use branched fully-connected layers to evaluate State Value V(s)and Advantage Value A(s,a) and final Q = V + A
* advantage value A(s,a) = Q(s,a) - Expect(Q(s,a)), where expectation of Q is also true V, by definition

## Policy-Based Learning
all we have learnt are value-based learning, where policies are made by values
now we can use NN to output possibility distribution of actions, to make our policy
* simple: no more values, especially for continous actions
* true stochastic: no argmax or fake random with epsilon, useful for game like rock-paper-sissors, and alias conditions (totally same condition, but need different actions)

### policy function approx.
just like value function approx.
policy func pi'(s,a,w) = Possibility p(a | s,w)
discrete action: softmax
continuous action: gaussian
* objective function J(w) = Expectation_pi of rewards tau, where tau is trejectory of rewards in the episode
tau can be expected state reward E(r) = sum(possibility(s,a)* R(s,a)), or can use start state value, G1, or average state/Q value V/Q

#### optimization
normal NN optimize on loss func, policy function optimize on objective function, hill climbing!
* stochastic policy search
randomly poking round and choose the ascending direction
pros: need no knowledge for objective function, policy function, just try states and policies
simulated annealing: start poking with high variance then gradually poke more carefully
adaptive noise scaling: when variance shrinked but no better policy if found, increase variance again to poke harder
* policy gradient
just like gradient descent, it is ascent on objective function by adjusting w, tricky math!
* constraint policy gradient
limit the difference between each gradient by threashold on deltha, or by adding a penalty term to the objective function
KL divergence to compare two distributions, in our case two policies


### Actor-Critic RL
use V/Q (either MC or TD) in place of tau to evaluate policy for objective function, and train the policy function
* a complete iteration
policy update pi(s,a,w_p) <-actor+critic-> value update q(s,a, w_q)

### utility
### stationry
### Bellman equation: policy iter and value iter

## Reinforcement Learning
### monte carlo learning

### temporal difference learning
for endless task
update reward at every step with next's reward

### learn to solve MDP without knowing TR
interact via state, action, transition

### Q learning: convergence, family, 
Q table: (Q for) quality of actions(column) for each state(row)
#### Bellman Equation
init Q table with 0
given policy, Q_func(state, action) = Expect(sum(gamma^i* reward_i) | state, action)
update Q table with epsilon greedy strategy
* epsilon = exploration rate, 1 as max means totally random
at each state we randomly generate a num, if greater than epsilon, we exploide, take action with known highest expected rewards
we gradually decrease epsilon, lower means more confidence
take action, reach state_1 and get reward_1, 
update Q(state_0, a) __Bellman Equation__
New Q(s_0,a) = Current Q value + learning_rate * 
   	[Reward(s_0,a) + discount_rate * (max(Q(s_1,actions)) — Q(s_0,a))]

### Deep Q Learning






### Exploration exploitation: learn and use, optimism
### RL approach: policy search, model-based RL
#### value based
optimize V(state), a function that tells us the maximum expected future reward the agent will get at each state
#### policy search
optimize p(state), a policy that maps state to best action
deterministic vs. stochastic: 
### Planning: function approx.

## DEEP rl
### deep Q learning


## Game Theory
mechnism design
representation of tree or matrix
minimax, maxmin
hidden
zero-sum
deterministic
pure/mix strategy
equilibrium, prisoner delimma
