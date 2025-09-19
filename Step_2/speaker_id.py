from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json, numpy as np
import soundfile as sf

try:
    from speechbrain.pretrained import EncoderClassifier  # type: ignore
    _SB_OK = True
except Exception:
    _SB_OK = False

def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-9
    return x / n

def _mfcc_embed(wav: np.ndarray, sr: int) -> np.ndarray:
    import python_speech_features as psf
    m = psf.mfcc(wav, sr, numcep=24)
    return _l2(m.mean(axis=0)).astype(np.float32)

@dataclass
class SpeakerDB:
    path: Path
    embeddings: Dict[str, List[List[float]]] = field(default_factory=dict)
    _centroids: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _matrix: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _names: List[str] = field(default_factory=list, init=False, repr=False)

    def load(self):
        if self.path.exists():
            self.embeddings = json.loads(self.path.read_text(encoding="utf-8"))
        self._rebuild_cache()
        return self

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.embeddings), encoding="utf-8")

    def _rebuild_cache(self):
        self._centroids.clear()
        self._names = []
        mats: List[np.ndarray] = []
        for name, lst in self.embeddings.items():
            if not lst:
                continue
            arr = np.array(lst, dtype=np.float32)
            c = _l2(arr.mean(axis=0))
            self._centroids[name] = c
            self._names.append(name)
            mats.append(c[None, :])
        self._matrix = np.vstack(mats) if mats else None

    def add(self, name: str, emb: np.ndarray):
        name = name.strip()
        self.embeddings.setdefault(name, [])
        self.embeddings[name].append(emb.tolist())
        self.save()
        self._rebuild_cache()

    def names_and_matrix(self) -> Tuple[List[str], Optional[np.ndarray]]:
        return self._names, self._matrix

class SpeakerID:
    def __init__(self, db_path: str, threshold: float = 0.65):
        self.db = SpeakerDB(Path(db_path)).load()
        self.threshold = threshold
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        ) if _SB_OK else None

    def _embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if self.model is not None:
            import torch  # type: ignore
            t = torch.tensor(wav).float().unsqueeze(0)
            with torch.no_grad():
                emb = self.model.encode_batch(t, normalize=True).squeeze(0).squeeze(0).cpu().numpy()
            return _l2(emb.astype(np.float32))
        return _mfcc_embed(wav, sr)

    def enroll(self, name: str, wav_path: str) -> None:
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        emb = self._embed(wav.astype("float32"), int(sr))
        self.db.add(name, emb)

    def identify(self, wav_path: str) -> Optional[str]:
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        dur = len(wav) / float(sr)
        if dur < 0.8:  # min_seconds gate to avoid junk IDs on interjections
            return None
        q = self._embed(wav.astype("float32"), int(sr))  # (D,)
        names, M = self.db.names_and_matrix()            # M: (N,D)
        if M is None or not names:
            return None
        sims = M @ q                                     # cosine (L2-normed)
        i = int(np.argmax(sims))
        best, best_score = names[i], float(sims[i])
        return best if best_score >= self.threshold else None

    def list_speakers(self) -> list[str]:
        return list(self.db._names)
