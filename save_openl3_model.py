import tensorflow as tf
import openl3
from pathlib import Path

# Where to save the model
save_path = Path("models/openl3")
save_path.mkdir(parents=True, exist_ok=True)

print("Loading OpenL3 model from internet (only first time)...")
model = openl3.models.load_audio_embedding_model(
    input_repr="mel256",
    content_type="music",
    embedding_size=512
)

print("Saving SavedModel format to:", save_path)
tf.saved_model.save(model, str(save_path))

print("🎉 OpenL3 model saved successfully!")
