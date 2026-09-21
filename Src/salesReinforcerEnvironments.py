import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf
import pickle
from imblearn.over_sampling import SMOTENC
from os.path import exists
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.feature_selection import chi2
from scipy.stats import spearmanr
import json
from sklearn.metrics import f1_score
from google.colab import drive
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/SalesReinforcer/Src/')
import dataPrep
import classicRF

class BaseSalesEnv(gym.Env):
  def __init__(self, data):
    super().__init__()
    self.data = data
    self.current_step = 0
    self.max_steps = 100
    self.typeName = "BaseSalesEnv"

  def reset(self, seed=None, options=None):
      super().reset(seed=seed)
      self.current_step = 0
      return self._get_observation(), {}

  def _get_observation(self):
      raise NotImplementedError

  def _calculate_reward(self):
      raise NotImplementedError

class FeatureSelectionEnv(BaseSalesEnv):
  def __init__(self, train_df, dev_df):
    # Note: We are no longer passing 'data' to the super().__init__ if it's not used there.
    # Assuming BaseSalesEnv does not need the full 'data' but rather 'train_df' and 'dev_df'
    # If BaseSalesEnv's init needs the combined data, you'll need to pass it or adapt BaseSalesEnv.
    super().__init__(train_df) # Passing train_df to BaseSalesEnv's __init__ for now
    self.typeName = "FeatureSelectionEnv"
    self.train = train_df
    self.dev = dev_df

    # Assuming 'isSubscribed' is the target column and should be excluded from features
    # It's better to explicitly list feature columns to avoid issues with target position
    all_columns = self.train.columns.tolist()
    self.target_column = 'isSubscribed'

    excluded = {
        "ID",
        "Country",
        "Education",
        "Status",
        "Stage",
        "isSubscribed"
    }

    feature_columns_list = [
        col for col in all_columns
        if col not in excluded
    ]

    self.n_initial_features = len(feature_columns_list)
    self.feature_mask = np.zeros(self.n_initial_features, dtype=np.int8)

    # Store original feature names for easy lookup
    self.original_feature_names = feature_columns_list

    # Load base model
    base_model = classicRF.retrieveModelFromDrive("baseRandomForest.pkl")

    # Hyperparameters for RandomForest
    self.n_estimators = base_model.n_estimators

    # Define options for max_features and set initial index
    # Separating fixed non-numerical options from numerical ones for robust sorting
    fixed_options = [None, "sqrt", "log2"]
    numerical_defaults = [0.25, 0.5, 0.75, 1.0]

    # Combine to form the initial full list of options, sorted numerically after fixed ones
    self.max_features_options = fixed_options + sorted(numerical_defaults)

    if base_model.max_features is None:
        self.max_features_idx = self.max_features_options.index(None) # Should be 0
    elif isinstance(base_model.max_features, str):
        self.max_features_idx = self.max_features_options.index(base_model.max_features)
    else: # assuming it's a float or int
        if base_model.max_features not in self.max_features_options:
            # If a new numerical max_features is encountered, add it and re-sort numerical part
            current_numerical_options = [x for x in self.max_features_options if isinstance(x, (float, int))]
            current_numerical_options.append(base_model.max_features)
            self.max_features_options = fixed_options + sorted(current_numerical_options)

        self.max_features_idx = self.max_features_options.index(base_model.max_features)

    self.criterion_options = ["gini", "entropy", "log_loss"]
    if base_model.criterion == 'gini':
        self.criterion_idx = 0
    elif base_model.criterion == 'entropy':
        self.criterion_idx = 1
    else:
        self.criterion_idx = 2 # log_loss

    self.max_depth = base_model.max_depth if base_model.max_depth is not None else 10 # Default if None in base model

    self.initial_n_estimators = self.n_estimators
    self.initial_max_features_idx = self.max_features_idx
    self.initial_criterion_idx = self.criterion_idx
    self.initial_max_depth = self.max_depth

    # Penalty for adding features
    self.feature_penalty_weight = 0.01 # Adjustable parameter

    # Define actions:
    # 0 to n_initial_features-1: Toggle feature i
    # n_initial_features: Increase n_estimators
    # n_initial_features + 1: Decrease n_estimators
    # n_initial_features + 2: Cycle max_features to next option
    # n_initial_features + 3: Cycle max_features to previous option
    # n_initial_features + 4: Cycle criterion to next option
    # n_initial_features + 5: Cycle criterion to previous option
    # n_initial_features + 6: Increase max_depth
    # n_initial_features + 7: Decrease max_depth
    self.action_space = spaces.Discrete(self.n_initial_features + 8)
    self.observation_space = spaces.Dict({
        "feature_mask": spaces.MultiBinary(self.n_initial_features),
        "n_estimators": spaces.Box(low=10,high=500,shape=(1,),dtype=np.int32),
        "max_features_idx": spaces.Discrete(len(self.max_features_options)),
        "criterion_idx": spaces.Discrete(len(self.criterion_options)),
        "max_depth": spaces.Box(low=1, high=50, shape=(1,), dtype=np.int32)
    })


  def _get_observation(self):
    return {
        "feature_mask": self.feature_mask.copy(),
        "n_estimators": np.array([self.n_estimators], dtype=np.int32),
        "max_features_idx": self.max_features_idx,
        "criterion_idx": self.criterion_idx,
        "max_depth": np.array([self.max_depth], dtype=np.int32)
    }

  def _calculate_reward(self):
    # Select features based on the mask
    selected_feature_indices = np.where(self.feature_mask == 1)[0]
    if len(selected_feature_indices) == 0:
        return 0.0 # No features selected, reward is 0

    # Get feature names from the training data using the stored original names
    selected_features = [self.original_feature_names[i] for i in selected_feature_indices]

    X_train = self.train[selected_features]
    y_train = self.train[self.target_column]
    X_dev = self.dev[selected_features]
    y_dev = self.dev[self.target_column]

    # Initialize model with current hyperparameters
    current_max_features = self.max_features_options[self.max_features_idx]
    current_criterion = self.criterion_options[self.criterion_idx]

    model = rf(
        n_estimators=self.n_estimators,
        max_features=current_max_features,
        criterion=current_criterion,
        max_depth=self.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_dev)
    reward = f1_score(y_dev, predictions)

    # Apply penalty for the number of selected features
    reward -= len(selected_feature_indices) * self.feature_penalty_weight

    return reward

  def step(self, action):
    self.current_step += 1
    terminated = False
    truncated = False

    if action < self.n_initial_features:
      # Toggle feature
      self.feature_mask[action] = 1 - self.feature_mask[action]
    elif action == self.n_initial_features:
      # Increase n_estimators
      self.n_estimators = min(self.n_estimators + 10, 500) # Cap at 500
    elif action == self.n_initial_features + 1:
      # Decrease n_estimators
      self.n_estimators = max(self.n_estimators - 10, 10) # Min at 10
    elif action == self.n_initial_features + 2:
      # Cycle max_features to next option
      self.max_features_idx = (self.max_features_idx + 1) % len(self.max_features_options)
    elif action == self.n_initial_features + 3:
      # Cycle max_features to previous option
      self.max_features_idx = (self.max_features_idx - 1 + len(self.max_features_options)) % len(self.max_features_options)
    elif action == self.n_initial_features + 4:
      # Cycle criterion to next option
      self.criterion_idx = (self.criterion_idx + 1) % len(self.criterion_options)
    elif action == self.n_initial_features + 5:
      # Cycle criterion to previous option
      self.criterion_idx = (self.criterion_idx - 1 + len(self.criterion_options)) % len(self.criterion_options)
    elif action == self.n_initial_features + 6:
      # Increase max_depth
      self.max_depth = min(self.max_depth + 1, 50) # Cap at 50
    elif action == self.n_initial_features + 7:
      # Decrease max_depth
      self.max_depth = max(self.max_depth - 1, 1) # Min at 1

    reward = self._calculate_reward()

    terminated = False
    truncated = self.current_step >= self.max_steps

    return (
        self._get_observation(),
        reward,
        terminated,
        truncated,
        {}
    )

  def reset(self, seed=None, options=None):
    gym.Env.reset(self, seed=seed)

    self.current_step = 0
    self.feature_mask[:] = 0
    self.n_estimators = self.initial_n_estimators
    self.max_features_idx = self.initial_max_features_idx
    self.criterion_idx = self.initial_criterion_idx
    self.max_depth = self.initial_max_depth

    return self._get_observation(), {}

class BusinessActionEnv(BaseSalesEnv):
  def __init__(self, data): # 'data' here is the training dataframe (train)
    super().__init__(data)
    self.typeName = "BusinessActionEnv"
    self.actions = [
        "call",
        "schedule_demo",
        "request_survey",
        "signup_platform",
        "assign_account_manager"
    ]

    self.action_space = spaces.Discrete(len(self.actions))

    self.target_column = 'isSubscribed'

    # Columns to explicitly exclude from observation features
    # These are likely original date strings, original categorical columns, or identifiers
    columns_to_exclude = ['ID', 'First Contact', 'Last Contact', 'First Call',
                          'Signed up for a demo', 'Filled in customer survey',
                          'Did sign up to the platform', 'Account Manager assigned',
                          'Country', 'Education', 'Status', 'Stage',
                          self.target_column]

    # Dynamically build the list of observation features from the dataframe's columns
    # This assumes the input `data` (train_df) already has the one-hot encoded columns etc.
    self.observation_features = [col for col in self.data.columns if col not in columns_to_exclude]

    num_observation_features = len(self.observation_features)
    # The low and high bounds for the Box space. Assuming features can be any float.
    # Modified to wrap the Box space into a Dict for MultiInputPolicy compatibility
    self.observation_space = spaces.Dict({
        "features": spaces.Box(low=-np.inf, high=np.inf, shape=(num_observation_features,), dtype=np.float32)
    })

    # Reward calculation:
    # `self.data` is the train dataframe. We iterate over its rows.
    self.current_customer_idx = 0
    self.n_customers = len(self.data)

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.current_customer_idx = 0
    return self._get_observation(), {}


  def _get_observation(self):
    if self.current_customer_idx >= self.n_customers:
        # This state should ideally be handled by 'terminated' flag. Return a dummy observation.
        # Return a dictionary matching the Dict observation space
        return {"features": np.zeros(len(self.observation_features), dtype=np.float32)}

    # Get the features for the current customer
    current_customer_row = self.data.iloc[self.current_customer_idx]
    observation = current_customer_row[self.observation_features].values.astype(np.float32)
    # Return a dictionary matching the Dict observation space
    return {"features": observation}

  def _calculate_reward(self):
    if self.current_customer_idx >= self.n_customers:
        return 0.0 # No more customers, no reward

    # For simplicity, reward is based on whether the current customer is subscribed.
    # This assumes we are evaluating the *final* outcome of the customer being observed.
    # A more complex environment would involve predicting subscription probability or
    # change in stage based on the action taken.
    current_customer_row = self.data.iloc[self.current_customer_idx]
    reward = float(current_customer_row[self.target_column]) # Convert boolean/int to float
    return reward

  def step(self, action):
    # For this simplified environment, an action doesn't change the customer's state within the environment.
    # We are just moving to the next customer and observing their outcome.
    # In a real Business Action RL, the 'action' would interact with a model
    # that simulates the impact of the action on the customer's state/probability of subscription.

    # Before getting observation for next step, calculate reward for current step (current customer)
    reward = self._calculate_reward()

    # Move to the next customer
    self.current_customer_idx += 1

    terminated = self.current_customer_idx >= self.n_customers
    truncated = False # Not truncating based on episode length for now

    # Get observation for the *next* state (next customer)
    if not terminated:
        observation = self._get_observation()
    else:
        # Terminal state observation, return a dictionary
        observation = {"features": np.zeros(len(self.observation_features), dtype=np.float32)} # Terminal state observation

    info = {} # No extra info for now

    return (
        observation,
        reward,
        terminated,
        truncated,
        info
    )

def testRawEnvironment(raw_env):
  check_env(raw_env)
  obs, info = raw_env.reset()
  print(obs)

  for _ in range(10):
      action = raw_env.action_space.sample()
      obs, reward, terminated, truncated, info = raw_env.step(action)
      print(action, reward, terminated, truncated)

      if terminated or truncated:
          obs, info = raw_env.reset()

def main():
  full_train_data = dataPrep.retrieveCSVFromDrive("SalesReinforcerTrain.csv")

  # Split the full training data into train and dev sets
  train, dev = train_test_split(full_train_data, test_size=0.2, random_state=42,stratify=full_train_data["isSubscribed"])

  # register environments
  register(id="SalesFeatureSelection-v0",entry_point=FeatureSelectionEnv)
  register(id="SalesBusinessAction-v0",entry_point=BusinessActionEnv)

  featureEnv = gym.make("SalesFeatureSelection-v0",train_df=train,dev_df=dev)
  businessActionEnv = gym.make("SalesBusinessAction-v0",data=train)

  testRawEnvironment(featureEnv)
  testRawEnvironment(businessActionEnv)

  stressTestSteps = 10000

  for env in [featureEnv, businessActionEnv]:
    model = DQN("MultiInputPolicy",
                env,
                verbose=1,
                seed=42)
    model.learn(total_timesteps=stressTestSteps)
    print(f'{env.unwrapped.typeName} survived {stressTestSteps} steps')




if __name__ == "__main__":
  main()
