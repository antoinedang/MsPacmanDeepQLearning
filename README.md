# How it works
The agent learns to make the best decision at each time step by constantly learning to map (state, action) pairs to reward expectations. The agent's state information included information about available collectible rewards, the proximity of ghosts, and the presence of walls limiting movement. The reward function was not that given directly by the MsPacman game (as these rewards were too sparse), instead it is a combination of the change in distance from ghosts and collectibles picked up. <br>
The agent ran on a simulation of MsPacman with OpenAI's Gym environment and learned the (state,action) -> reward function with a fully connected Neural Network built in PyTorch.

# Results
The agent is able to very reliably beat level 1 (4000 points), and the highest recorded score is about 12000 points. During the in-class competition, a score of over 10000 points was recorded.

# Try it yourself
**To set up dependencies:** <br>
    Install Python 3<br>
    Run `pip install -r requirements.txt`

**To run agent:**
    `python3 play.py`

This will load the best performing agent saved in data/checkpoints/best_agent.pkl. <br>
A random seed will be used every time `play.py` is run. To set the seed manually, modify line *14* of `play.py`
