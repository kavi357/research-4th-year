import crepe
from pathlib import Path

Path("models").mkdir(exist_ok=True)

model = crepe.core.build_and_load_model(model_capacity='small')
model.save("models/crepe_small.keras")

print("CREPE Keras model saved")
