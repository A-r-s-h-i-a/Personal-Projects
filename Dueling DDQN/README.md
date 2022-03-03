# Dueling Double Deep Q Network
Deep Q Networks (DQN) use a Neural Network (NN) to learn a policy, replacing "Q Tables" in the original Q Learning algorithm. This approach enables agents to operate in continuous environments where large amounts of input data must be processed for any hope of good performance. A DQN's network takes a state as input and gives as output scores for all available actions. The DQN then chooses the action with the highest score to perform. A Dueling DQN, however, also takes into account an action's Advantage when calculating scores, not just the current state. The Advantage is how much better the action is (positively or negatively) than all other actions. Thus, in a Dueling DQN, bad actions have much worse scores, and good actions much stronger ones.

# Results

![Training Results](https://github.com/A-r-s-h-i-a/Personal-Projects/blob/main/Dueling%20DDQN/Dueling%20Double%20DQN%20Performance%202.png)

Untrained Model:

https://user-images.githubusercontent.com/46332063/156497697-18e04ef1-fce8-45ff-8dba-86768b5911ff.mp4

Trained Model:

https://user-images.githubusercontent.com/46332063/156497684-6e5ef31c-3f56-4896-a569-f1c491adb114.mp4

