from dask.distributed import Client
# Library to generate plots                                                                         
import matplotlib as mpl                                                                            
# Define Agg as Backend for matplotlib when no X server is running                                  
mpl.use('Agg')                                                                                      
import matplotlib.pyplot as plt
import dask
import xgboost as xgb
import dask.array as da
import json
import os

data=[]
# Using the distributed shared file system, we can access to the Dask cluster
# configuration.
# We read the scheduler address and port from the scheduler file
with open(os.environ["SCHEDULER_FILE"]) as f:
        data = json.load(f)
        scheduler_address=data['address']

# Connect to the the cluster
client = Client(scheduler_address)
client.wait_for_workers()

# X and y must be Dask dataframes or arrays
num_obs = 1e5
num_features = 20
X = da.random.random(size=(num_obs, num_features), chunks=(1000, num_features))
y = da.random.random(size=(num_obs, 1), chunks=(1000, 1))

# Training
dtrain = xgb.dask.DaskDMatrix(client, X, y)

output = xgb.dask.train(
    client,
    {"verbosity": 2, "tree_method": "hist", "objective": "reg:squarederror"},
    dtrain,
    num_boost_round=10,
    evals=[(dtrain, "train")],
)

booster = output['booster']  # booster is the trained model
history = output['history']  # A dictionary containing evaluation results

ax = xgb.plot_importance(booster, height=0.8, max_num_features=9)
ax.grid(False, axis="y")
ax.set_title('Estimated feature importance')
plt.savefig("importance.png")

# Stop Dask cluster
client.shutdown()

