import json
from audit_cached import DEST, SEQS, vio

result = {}
for seq in SEQS:
    source = DEST/'artifacts'/('rerun_'+seq)
    process = json.loads((source/'process.json').read_text(encoding='utf-8-sig'))
    if process['exit_code'] != 0:
        raise RuntimeError(f'{seq}: process failed, inspect run.log')
    log = (source/'run.log').read_text(encoding='utf-16', errors='replace')
    # Windows PowerShell redirection writes UTF-16 logs. roslaunch may exit 0
    # even when a required child node aborted: never equate container exit with health.
    if 'Setting printing' not in log:
        log = (source/'run.log').read_text(encoding='utf-8-sig', errors='replace')
    process['required_node_abort'] = 'exit code -6' in log
    process['library_unload_exception'] = 'LibraryUnloadException' in log
    result[seq] = dict(process=process, metrics=vio(seq, source))
(DEST/'artifacts/openvins_rerun_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({s: dict(n=a['metrics']['n_matched'], ATE_m=a['metrics']['ATE_rmse_m'], wall_s=a['process']['wall_s']) for s,a in result.items()}, indent=2))
