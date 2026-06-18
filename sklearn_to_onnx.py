import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

clf = joblib.load("blink_model.pkl")
scaler = joblib.load("scaler.pkl")

# Combine into a pipeline
pipeline = Pipeline([
    ("scaler", scaler),
    ("classifier", clf),
])

#number of feautures = 6
initial_type = [("input", FloatTensorType([None, 6]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

# Save
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())