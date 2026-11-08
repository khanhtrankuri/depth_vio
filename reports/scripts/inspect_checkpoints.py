"""Validate local tensor checkpoints without instantiating/downloading networks."""
import hashlib
import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[2]
result = {'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'checkpoints': {}}
for model in ['igev_plusplus', 'igev_rt']:
    p = root/'IGEV-plusplus/pretrained_models'/model/'sceneflow.pth'
    state = torch.load(p, map_location='cpu', weights_only=True)
    result['checkpoints'][model] = dict(
        path=str(p.relative_to(root)), bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        tensor_entries=len(state), parameter_and_buffer_elements=sum(t.numel() for t in state.values()),
        tensors_finite=all(torch.isfinite(t).all().item() for t in state.values()),
        sample_keys=list(state)[:5],
        status='readable finite tensor state_dict; strict architecture load and inference NOT EVALUATED')
(root/'reports/artifacts/checkpoints.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
