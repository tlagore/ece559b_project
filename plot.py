import matplotlib.pyplot as plt 
import pandas as pd
import pickle
from agent import TestCase, AgentConfiguration
from snake import StateAttributeType

def plot_data(df: list[tuple[pd.DataFrame, dict]], x_range: tuple[int, int]):

    _, axes = plt.subplots(nrows=1, ncols=len(df), figsize=(16, 4))
    if len(df) > 1:
        for i in range(len(df)):
            df[i][0].plot(ax=axes[i], **df[i][1])
            axes[i].set_xlim(x_range[0], x_range[1])
    else:
        df[0][0].plot(ax=axes, **df[0][1])
        axes.set_xlim(x_range[0], x_range[1])
    
    plt.tight_layout()
    plt.show()

with open('results/state-features_single-network.pickle', 'rb') as in_file:
    obj:TestCase = pickle.load(in_file)

# args = {
#     'title': f"<test_case_placeholder>", 
#     'xlabel': "Episodes", 
#     'ylabel': "Scores"
# }

train_df = pd.DataFrame(obj.training_rewards)
test_df = pd.DataFrame(obj.test_rewards)

_, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
train_df.plot.scatter(ax=axes[0], x='episodes', y='scores')
test_df.plot.scatter(ax=axes[1], x='episodes', y='scores')

plt.tight_layout()
plt.show()