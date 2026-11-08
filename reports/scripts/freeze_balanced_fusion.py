"""Finalize the user-requested balanced tradeoff using development results only."""
import json
import re
from evaluate_fusion_v2 import DEST, ROOT, choose_balanced, sha, write_json

source=DEST/'V1_01_easy.json'; data=json.loads(source.read_text()); results=data['results']
for seq in ['V1_02_medium','V1_03_difficult']:
    if (DEST/(seq+'.json')).exists():
        raise RuntimeError('Cannot change the preset after held-out results were produced')
selected,eligible,best=choose_balanced(results)
selection=DEST/'selection.json'
if selection.exists(): raise RuntimeError('Preserve earlier development-only selection before finalizing')
extract=eligible[selected]['extraction']
choice=dict(development_sequence='V1_01_easy',selected=selected,variant=selected.rsplit('_support',1)[0],
            min_surface_support=extract.get('min_surface_support',int(re.search(r'_support(\d+)',selected).group(1))),
            min_component_points=extract.get('min_component_points',0),min_component_faces=extract.get('min_component_faces',30),
            criterion='fastest integration within 0.5pp of best development F1@5cm; recall no more than 2pp below native uniform/support0; ties use higher F1',
            best_development_f1=best,heldout_sequences=['V1_02_medium','V1_03_difficult'],
            history='Development candidate search expanded before any held-out metrics; quality-only selector replaced by balanced selector to honor user priority. Earlier selections preserved.',
            result_sha256=sha(source),code_sha256={p.name:sha(p) for p in (ROOT/'pipeline').glob('fusion_*.py')})
write_json(selection,choice)
write_json(ROOT/'pipeline/fusion_balanced.json',dict(schema_version=1,variant=choice['variant'],
           min_surface_support=choice['min_surface_support'],min_component_points=choice['min_component_points'],
           min_component_faces=choice['min_component_faces'],voxel_m=.03,trunc_m=.12,zmax_m=4.,
           point_surface_not_mesh=choice['variant'].startswith('surfel'),selected_on='V1_01_easy',
           selection_sha256=sha(selection)))
print(json.dumps(choice,indent=2))
