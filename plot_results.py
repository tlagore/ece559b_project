import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import pickle
from agent import TestCase, AgentConfiguration
from snake import StateAttributeType
from pprint import pprint

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
    single:TestCase = pickle.load(in_file)

with open('results/state-features_target-network.pickle', 'rb') as in_file:
    target:TestCase = pickle.load(in_file)

# pprint(single.config)
# exit()

train_df = pd.DataFrame(single.training_rewards)
test_df = pd.DataFrame(single.test_rewards)

train_target_df = pd.DataFrame(target.training_rewards)
test_target_df = pd.DataFrame(target.test_rewards)

train_df = train_df.join(train_target_df['scores'], rsuffix='_target')

test_df = test_df.join(test_target_df['scores'], rsuffix='_target')

print(train_df.info())
print(train_df.describe())

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,8))
sns.regplot(train_df['episodes'],train_df['scores'], color='red', label='Single Network', ax=axes[0])
ax = sns.regplot(train_df['episodes'],train_df['scores_target'], color='green', label='Target Network', ax=axes[0])

ax.set(
    xlabel="Episodes",
    ylabel="Score")
ax.set_title('Training over 120 episodes')
ax.legend()

sns.regplot(test_df['episodes'],test_df['scores'], color='red', label='Single Network', ax=axes[1])
ax = sns.regplot(test_df['episodes'],test_df['scores_target'], color='green', label='Target Network', ax=axes[1])
ax.set(
    xlabel="Episodes",
    ylabel="Score")
ax.set_title('Test Performance over 10 episodes')

ax.legend()
# plt.suptitle("More Short-Lived Episodes | Lower Batch Size")
plt.tight_layout()
plt.show()

# _, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
# train_df.plot.scatter(ax=axes[0], x='episodes', y='scores', color='red')
# train_df.plot.scatter(ax=axes[0], x='episodes', y='scores_target', color='green')
# test_df.plot.scatter(ax=axes[1], x='episodes', y=['scores', 'scores_target'])

# plt.tight_layout()
# plt.show()