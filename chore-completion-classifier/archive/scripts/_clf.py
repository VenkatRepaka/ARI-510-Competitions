import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

CLASSES = ["Washing Dishes|Complete", "Washing Dishes|Incomplete", "Taking Out Trash|Complete", "Taking Out Trash|Incomplete"]
train = pd.read_csv("data/train.csv"); test = pd.read_csv("data/test.csv")
y = train.label.map(CLASSES.index).values
unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
M = sys.argv[1].split(",") if len(sys.argv) > 1 else ["vitl14_openai", "vith14_laion", "siglip_so400m", "eva02_l"]
F = {}
for n in M:
    z = np.load(f"features/{n}.npz"); pos = {i: k for k, i in enumerate(z["ids"])}
    tr = [pos[i] for i in train.id]; te = [pos[i] for i in test.id]
    both = unit(z["img"] + z["img_flip"]); m7 = unit(z["img"] + z["img_flip"] + z["crops"].sum(1))
    F[n] = {"both": (both[tr], both[te]), "mean7": (m7[tr], m7[te])}

clfs = {
    "LR C=1": lambda: LogisticRegression(C=1, max_iter=5000, class_weight="balanced"),
    "LR C=10": lambda: LogisticRegression(C=10, max_iter=5000, class_weight="balanced"),
    "LR std C=0.1": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")),
    "linSVM C=1": lambda: SVC(kernel="linear", C=1, probability=False, class_weight="balanced"),
    "rbfSVM C=10": lambda: SVC(kernel="rbf", C=10, class_weight="balanced"),
    "ridge a=1": lambda: RidgeClassifier(alpha=1, class_weight="balanced"),
    "LDA shrink": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "PCA32+LDA": lambda: make_pipeline(PCA(32), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "centroid": lambda: NearestCentroid(),
    "kNN5 cos": lambda: KNeighborsClassifier(5, metric="cosine", weights="distance"),
}
splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0).split(y, y))

def cv(X, mk):
    return np.mean([f1_score(y[va], mk().fit(X[tr], y[tr]).predict(X[va]), average="macro") for tr, va in splits])

def cv_spread(X, Xte, k=7, alpha=0.2):
    # transductive: unlabeled test photos join the graph
    s = []
    for tr, va in splits:
        Xa = np.vstack([X, Xte]); ya = -np.ones(len(Xa), int); ya[tr] = y[tr]
        ls = LabelSpreading(kernel="knn", n_neighbors=k, alpha=alpha, max_iter=200).fit(Xa, ya)
        s.append(f1_score(y[va], ls.transduction_[va], average="macro"))
    return np.mean(s)

rows = []
sets = {n: n for n in M}
sets["concat_all"] = None
for view in ("both", "mean7"):
    for sname in sets:
        if sets[sname]:
            X, Xte = F[sname][view]
        else:
            X = np.hstack([unit(F[n][view][0]) for n in M]); Xte = np.hstack([unit(F[n][view][1]) for n in M])
        r = {"features": sname, "view": view}
        for c, mk in clfs.items():
            r[c] = cv(X, mk)
        r["labelSpread"] = cv_spread(X, Xte)
        rows.append(r); print(sname, view, flush=True)
df = pd.DataFrame(rows).round(3)
pd.set_option("display.width", 250)
print(df.to_string(index=False))
