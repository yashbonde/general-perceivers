import os
os.environ['GPERC_MODE'] = 'client'
os.environ["GPERC_URL"] = "http://192.168.0.196:6969"

def hr():
  print("-" * 80)

from orchestration import create_model, status, train_model

# step 1: create a model and check the status from orchestrators point of view
model = create_model(latent_dim = 8, modulo = 10, max_len = 2)
print("model_id:", model["model_id"])
orc_status = status()
print("n_models:", orc_status["n_models"])

# step 2: tell the model to start training
out = train_model(model["model_id"], "_plus_op", num_steps = 1000)
print(out)
