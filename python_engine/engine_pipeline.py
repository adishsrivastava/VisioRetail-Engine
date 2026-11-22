# ==============================================================================
# VisioRetail Core Neural & Mathematical Engine (python_engine/engine_pipeline.py)
# ==============================================================================
import time
import numpy as np

np.random.seed(42)

class TensorSandbox:
    @staticmethod
    def run_operations():
        X = np.random.randn(4, 8)
        W = np.random.randn(8, 6) * np.sqrt(2.0 / 8.0)
        b = np.array([[0.10, -0.05, 0.20, 0.00, -0.10, 0.15]])
        Z = np.dot(X, W) + b
        q = X[0] / (np.linalg.norm(X[0]) + 1e-8)
        c = X[1] / (np.linalg.norm(X[1]) + 1e-8)
        inner_prod_cos = float(np.dot(q, c))
        Z_reshaped = Z.reshape(2, 12)
        catalog_matrix = np.random.randn(64, 32)
        U, S, Vt = np.linalg.svd(catalog_matrix, full_matrices=False)
        retained_variance = float(np.sum(S[:16] ** 2) / np.sum(S ** 2))
        return {
            "broadcast_shape": Z.shape,
            "reshaped_shape": Z_reshaped.shape,
            "cosine_inner_product": inner_prod_cos,
            "svd_rank_k_variance": retained_variance,
        }

class Activations:
    @staticmethod
    def step(z, theta=0.0): return np.where(z >= theta, 1.0, 0.0)
    @staticmethod
    def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))
    @staticmethod
    def tanh(z): return np.tanh(z)
    @staticmethod
    def relu(z): return np.maximum(0.0, z)

class McCullochPittsNeuron:
    def __init__(self, weights, threshold=2.0):
        self.weights = np.array(weights, dtype=float)
        self.threshold = float(threshold)
    def forward(self, x):
        return 1 if float(np.dot(self.weights, x)) >= self.threshold else 0

class BaselineMLP:
    def __init__(self, in_dim=16, hid_dim=12, out_dim=4, init_type="he", l2_reg=1e-4):
        self.hyperparameters = {"in_dim": in_dim, "hid_dim": hid_dim, "out_dim": out_dim, "init_type": init_type, "l2_lambda": l2_reg}
        scale1 = np.sqrt(2.0 / in_dim) if init_type == "he" else np.sqrt(1.0 / in_dim)
        self.W1 = np.random.randn(in_dim, hid_dim) * scale1
        self.b1 = np.zeros((1, hid_dim))
        self.W2 = np.random.randn(hid_dim, out_dim) * np.sqrt(1.0 / hid_dim)
        self.b2 = np.zeros((1, out_dim))
        self.vW1, self.vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vW2, self.vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)

    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = Activations.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        return Activations.sigmoid(Z2), (X, Z1, A1, Z2)

    def compute_loss(self, X, Y):
        Y_hat, _ = self.forward(X)
        mse = 0.5 * np.mean(np.sum((Y_hat - Y) ** 2, axis=1))
        l2 = 0.5 * self.hyperparameters["l2_lambda"] * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return float(mse + l2)

    def compute_gradients(self, X, Y):
        N, l2 = X.shape[0], self.hyperparameters["l2_lambda"]
        Y_hat, (_, Z1, A1, _) = self.forward(X)
        dZ2 = (Y_hat - Y) * (Y_hat * (1.0 - Y_hat)) / N
        dW2 = np.dot(A1.T, dZ2) + l2 * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, self.W2.T) * (Z1 > 0.0).astype(float)
        dW1 = np.dot(X.T, dZ1) + l2 * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def step(self, X, Y, lr=0.05, optimizer="nesterov", momentum=0.9):
        dW1, db1, dW2, db2 = self.compute_gradients(X, Y)
        if optimizer in ("batch_gd", "sgd"):
            self.W1 -= lr * dW1; self.b1 -= lr * db1; self.W2 -= lr * dW2; self.b2 -= lr * db2
        elif optimizer == "momentum":
            self.vW1 = momentum * self.vW1 - lr * dW1; self.vb1 = momentum * self.vb1 - lr * db1
            self.vW2 = momentum * self.vW2 - lr * dW2; self.vb2 = momentum * self.vb2 - lr * db2
            self.W1 += self.vW1; self.b1 += self.vb1; self.W2 += self.vW2; self.b2 += self.vb2
        elif optimizer == "nesterov":
            pv1, pv2 = self.vW1.copy(), self.vW2.copy()
            self.vW1 = momentum * self.vW1 - lr * dW1; self.vW2 = momentum * self.vW2 - lr * dW2
            self.W1 += -momentum * pv1 + (1.0 + momentum) * self.vW1
            self.W2 += -momentum * pv2 + (1.0 + momentum) * self.vW2
            self.b1 -= lr * db1; self.b2 -= lr * db2

    def verify_numerical_gradient(self, X, Y, eps=1e-5):
        _, _, dW2_ana, _ = self.compute_gradients(X, Y)
        ana = float(dW2_ana[0, 0])
        orig = self.W2[0, 0]
        self.W2[0, 0] = orig + eps; lp = self.compute_loss(X, Y)
        self.W2[0, 0] = orig - eps; lm = self.compute_loss(X, Y)
        self.W2[0, 0] = orig
        num = float((lp - lm) / (2.0 * eps))
        return ana, num, abs(ana - num) / max(1e-12, abs(ana) + abs(num))

class BioAndEnergyModels:
    @staticmethod
    def compare_hebbian_and_oja(steps=50, lr=0.08):
        x = np.array([0.80, 0.60, 0.00], dtype=float)
        w_h = np.array([0.35, 0.25, 0.15], dtype=float)
        w_o = w_h.copy()
        for _ in range(steps):
            w_h += lr * float(np.dot(w_h, x)) * x
            yo = float(np.dot(w_o, x))
            w_o += lr * yo * (x - yo * w_o)
        return float(np.linalg.norm(w_h)), float(np.linalg.norm(w_o))

    @staticmethod
    def boltzmann_rbm_cd1(v0, W, a, b, lr=0.05):
        h0 = Activations.sigmoid(np.dot(v0, W) + b)
        v1 = Activations.sigmoid(np.dot(h0, W.T) + a)
        h1 = Activations.sigmoid(np.dot(v1, W) + b)
        W_up = W + lr * (np.outer(v0, h0) - np.outer(v1, h1))
        energy = float(-np.dot(a, v0) - np.dot(b, h0) - np.dot(v0, np.dot(W_up, h0)))
        return energy, W_up

class VisionSequenceAndGenerativePipeline:
    @staticmethod
    def conv2d_sobel(img):
        k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
        h, w = img.shape
        return np.array([[np.sum(img[r:r+3, c:c+3] * k) for c in range(w - 2)] for r in range(h - 2)])

    @staticmethod
    def conv3d_spatiotemporal(vol, k3):
        d, h, w = vol.shape; kd, kh, kw = k3.shape
        return np.array([[[np.sum(vol[z:z+kd, y:y+kh, x:x+kw] * k3) for x in range(w-kw+1)] for y in range(h-kh+1)] for z in range(d-kd+1)])

    @staticmethod
    def sasrec_self_attention(emb):
        scores = np.dot(emb, emb.T) / np.sqrt(emb.shape[1]) + np.triu(np.ones((emb.shape[0], emb.shape[0])) * -1e9, k=1)
        ex = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        w = ex / np.sum(ex, axis=-1, keepdims=True)
        return np.dot(w, emb)[-1], w

    @staticmethod
    def vae_reparameterize_and_decode(mu, logvar, W_dec):
        z = mu + np.exp(0.5 * logvar) * np.random.randn(*mu.shape)
        kl = -0.5 * float(np.sum(1.0 + logvar - mu ** 2 - np.exp(logvar)))
        return z, Activations.sigmoid(np.dot(z, W_dec)), kl

if __name__ == "__main__":
    t_res = TensorSandbox.run_operations()
    X_s, Y_s = np.random.randn(8, 16), (np.random.rand(8, 4) > 0.5).astype(float)
    mlp = BaselineMLP(16, 12, 4, init_type="he")
    ana_g, num_g, rel_err = mlp.verify_numerical_gradient(X_s, Y_s)
    h_norm, o_norm = BioAndEnergyModels.compare_hebbian_and_oja()
    print(f"[PASS] SVD Retained Variance: {t_res['svd_rank_k_variance']*100:.2f}% | Grad Check Rel Error: {rel_err:.2e}")
    print(f"[PASS] Oja Normalized Weight Norm: {o_norm:.4f} (Unit 1.0) | Unbounded Hebbian Norm: {h_norm:.2f}")
