
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import io
import os
import uuid
import threading
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "app", "models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# APP
# ============================================================


# Async prediction jobs
prediction_jobs = {}
prediction_lock = threading.Lock()

def _run_prediction_job(job_id, image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        features_8d = image_to_8d(image)

        with torch.no_grad():
            logits = hybrid(features_8d)
            probs = torch.softmax(logits, dim=1)[0]
            malignant_prob = float(probs[1].cpu())

        if malignant_prob >= 0.70:
            risk_level = "High"
        elif malignant_prob >= 0.40:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        result = {
            "success": True,
            "prediction": "malignant" if malignant_prob >= 0.5 else "benign",
            "risk_level": risk_level,
            "probabilities": {
                "benign": float(probs[0].cpu()),
                "malignant": malignant_prob
            },
            "model": "EfficientNet-B3 + 8-Qubit VQC",
            "quantum_features": features_8d[0].detach().cpu().tolist(),
            "disclaimer": "Research prototype for decision support only. Not a clinical diagnosis."
        }

        with prediction_lock:
            prediction_jobs[job_id] = {"status": "completed", "result": result}

    except Exception as e:
        with prediction_lock:
            prediction_jobs[job_id] = {
                "status": "failed",
                "error": str(e)
            }

app = FastAPI(
    title="PROJECT QUANTUM API",
    description="Hybrid Quantum-Classical Early Disease Detection Platform",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# EFFICIENTNET-B3
# ============================================================

b3 = efficientnet_b3(weights=None)

b3.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1536, 2)
)

b3_checkpoint = torch.load(
    os.path.join(
        MODEL_DIR,
        "efficientnet_b3_backbone.pth"
    ),
    map_location=device,
    weights_only=False
)

if "model_state_dict" in b3_checkpoint:
    b3.load_state_dict(b3_checkpoint["model_state_dict"])
else:
    b3.load_state_dict(b3_checkpoint)

b3.classifier = nn.Identity()

b3 = b3.to(device)
b3.eval()


# ============================================================
# SCALER
# ============================================================

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "cnn_feature_scaler.pkl"
    )
)


# ============================================================
# SUPERVISED PROJECTION
# 1536 → 128 → 32 → 8
# ============================================================

projection = nn.Sequential(
    nn.Linear(1536, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(128, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),

    nn.Linear(32, 8)
)

projection_checkpoint = torch.load(
    os.path.join(
        MODEL_DIR,
        "supervised_projection_8d_NEW.pth"
    ),
    map_location=device,
    weights_only=False
)

if "model_state_dict" in projection_checkpoint:
    projection.load_state_dict(
        projection_checkpoint["model_state_dict"]
    )
else:
    projection.load_state_dict(projection_checkpoint)

projection = projection.to(device)
projection.eval()


# ============================================================
# SAFE VQC
# ============================================================

class SafeVQC(nn.Module):

    def __init__(self, n_qubits=8, n_layers=3):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.q_weights = nn.Parameter(
            torch.randn(
                n_layers,
                n_qubits,
                2
            ) * 0.05
        )

        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

    def ry(self, theta):

        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)

        return torch.stack([
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1)
        ], dim=-2).to(torch.complex64)

    def rz(self, theta):

        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)

        zero = torch.zeros_like(c)

        return torch.stack([
            torch.stack([
                torch.complex(c, -s),
                zero
            ], dim=-1),

            torch.stack([
                zero,
                torch.complex(c, s)
            ], dim=-1)
        ], dim=-2)

    def _apply_single_qubit(self, state, gate, qubit):

        batch = state.shape[0]

        tensor = state.reshape(
            batch,
            *([2] * self.n_qubits)
        )

        tensor = tensor.movedim(
            qubit + 1,
            -1
        )

        tensor = torch.matmul(
            tensor,
            gate.transpose(-1, -2)
        )

        tensor = tensor.movedim(
            -1,
            qubit + 1
        )

        return tensor.reshape(batch, -1)

    def _apply_cnot(self, state, control, target):

        result = state.clone()

        dim = 2 ** self.n_qubits

        for i in range(dim):

            if ((i >> control) & 1) == 1:

                flipped = i ^ (1 << target)

                if i < flipped:

                    a = state[:, i].clone()
                    b = state[:, flipped].clone()

                    result[:, i] = b
                    result[:, flipped] = a

        return result

    def forward(self, x):

        batch = x.shape[0]

        state = torch.zeros(
            batch,
            2 ** self.n_qubits,
            dtype=torch.complex64,
            device=x.device
        )

        state[:, 0] = 1.0

        # Angle embedding
        for q in range(self.n_qubits):

            state = self._apply_single_qubit(
                state,
                self.ry(x[:, q]),
                q
            )

        # Variational layers
        for layer in range(self.n_layers):

            for q in range(self.n_qubits):

                state = self._apply_single_qubit(
                    state,
                    self.ry(
                        self.q_weights[
                            layer, q, 0
                        ].expand(batch)
                    ),
                    q
                )

                state = self._apply_single_qubit(
                    state,
                    self.rz(
                        self.q_weights[
                            layer, q, 1
                        ].expand(batch)
                    ),
                    q
                )

            # CNOT ring
            for q in range(self.n_qubits):

                state = self._apply_cnot(
                    state,
                    q,
                    (q + 1) % self.n_qubits
                )

        probs = torch.abs(state) ** 2

        expectations = []

        for q in range(self.n_qubits):

            values = torch.tensor(
                [
                    1.0
                    if ((i >> q) & 1) == 0
                    else -1.0
                    for i in range(
                        2 ** self.n_qubits
                    )
                ],
                device=x.device
            )

            expectations.append(
                torch.sum(
                    probs * values,
                    dim=1
                )
            )

        z = torch.stack(
            expectations,
            dim=1
        )

        return self.classifier(z)


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.vqc = SafeVQC(
            n_qubits=8,
            n_layers=3
        )

        # NOTE: checkpoint uses the name "classical"
        self.classical = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(16, 2)
        )

        self.fusion = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(16, 2)
        )

    def forward(self, x):

        q_logits = self.vqc(x)

        c_logits = self.classical(x)

        fusion_input = torch.cat(
            [q_logits, c_logits],
            dim=1
        )

        return self.fusion(fusion_input)


# ============================================================
# LOAD FINAL HYBRID CHECKPOINT
# ============================================================

hybrid = HybridModel()

hybrid_checkpoint = torch.load(
    os.path.join(
        MODEL_DIR,
        "hybrid_safe_vqc_FINAL.pth"
    ),
    map_location=device,
    weights_only=False
)

if "model_state_dict" in hybrid_checkpoint:
    hybrid.load_state_dict(
        hybrid_checkpoint["model_state_dict"]
    )
else:
    hybrid.load_state_dict(
        hybrid_checkpoint
    )

hybrid = hybrid.to(device)
hybrid.eval()


# ============================================================
# IMAGE → 8D
# ============================================================

def image_to_8d(image):

    image = image.convert("RGB")

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        features = b3(x)

        scaled = scaler.transform(
            features.cpu().numpy()
        )

        scaled = torch.tensor(
            scaled,
            dtype=torch.float32,
            device=device
        )

        features_8d = projection(scaled)

    return features_8d


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "project": "PROJECT QUANTUM",
        "problem": "SIH26139",
        "model": "Hybrid B3 + 8-Qubit VQC",
        "device": str(device)
    }


# ============================================================
# MODELS
# ============================================================

@app.get("/api/models")
def models():

    return {
        "models": [

            {
                "id": "hybrid_vqc",
                "name": "EfficientNet-B3 + 8-Qubit VQC",
                "type": "Hybrid Quantum-Classical",
                "qubits": 8,
                "layers": 3,
                "accuracy": 0.7821,
                "sensitivity": 0.7773,
                "specificity": 0.7870,
                "f1": 0.7826,
                "roc_auc": 0.8420
            },

            {
                "id": "efficientnet_b0",
                "name": "EfficientNet-B0",
                "type": "Classical CNN",
                "accuracy": 0.7179,
                "sensitivity": 0.7682,
                "specificity": 0.6667,
                "f1": 0.7332,
                "roc_auc": 0.7663
            },

            {
                "id": "resnet50",
                "name": "ResNet-50",
                "type": "Classical CNN",
                "accuracy": 0.6514,
                "sensitivity": 0.6318,
                "specificity": 0.6713,
                "f1": 0.6465,
                "roc_auc": 0.7139
            },

            {
                "id": "yolo26n",
                "name": "YOLO26n",
                "type": "Object Detection",
                "precision": 0.5000,
                "recall": 1.0000,
                "map50": 0.5351,
                "map50_95": 0.5351
            }
        ]
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        features_8d = image_to_8d(image)

        with torch.no_grad():

            logits = hybrid(
                features_8d
            )

            probabilities = F.softmax(
                logits,
                dim=1
            )[0]

        benign_prob = float(
            probabilities[0].item()
        )

        malignant_prob = float(
            probabilities[1].item()
        )

        prediction = (
            "malignant"
            if malignant_prob >= 0.5
            else "benign"
        )

        if malignant_prob >= 0.70:
            risk_level = "High"

        elif malignant_prob >= 0.40:
            risk_level = "Moderate"

        else:
            risk_level = "Low"

        return {

            "success": True,

            "prediction": prediction,

            "risk_level": risk_level,

            "probabilities": {
                "benign": benign_prob,
                "malignant": malignant_prob
            },

            "model": "EfficientNet-B3 + 8-Qubit VQC",

            "quantum_features": [
                float(v)
                for v in features_8d[
                    0
                ].detach().cpu().numpy()
            ],

            "disclaimer": (
                "Research prototype for decision support only. "
                "Not a clinical diagnosis."
            )
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


from fastapi import BackgroundTasks

@app.post("/api/predict-async")
async def predict_async(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    image_bytes = await file.read()

    job_id = str(uuid.uuid4())

    with prediction_lock:
        prediction_jobs[job_id] = {"status": "processing"}

    background_tasks.add_task(_run_prediction_job, job_id, image_bytes)

    return {
        "success": True,
        "job_id": job_id,
        "status": "processing"
    }


@app.get("/api/predict-status/{job_id}")
async def predict_status(job_id: str):
    with prediction_lock:
        job = prediction_jobs.get(job_id)

    if job is None:
        return {"success": False, "status": "not_found"}

    return {
        "success": True,
        "job_id": job_id,
        **job
    }
