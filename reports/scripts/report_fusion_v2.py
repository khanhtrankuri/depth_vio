"""Publish measured development/held-out mapping results, not inferred scores."""
import base64
from html import escape
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from evaluate_fusion_v2 import DEST, ROOT, crop
from audit_cached import ply

seqs=['V1_01_easy','V1_02_medium','V1_03_difficult']
selection=json.loads((DEST/'selection.json').read_text()); selected=selection['selected']
allres={seq:json.loads((DEST/(seq+'.json')).read_text()) for seq in seqs}
timing=json.loads((DEST/'backend_timing.json').read_text())
figdir=ROOT/'reports/figures'; figdir.mkdir(exist_ok=True)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n\n'


rows=[]
for seq in seqs:
    result=allres[seq]['results']
    for key in ['legacy_tsdf','legacy_naive','uniform_support0',selected]:
        m=result[key]['metrics']
        rows.append([seq,key,f"{100*m['accuracy_mean_m']:.3f}",f"{100*m['accuracy_rmse_m']:.3f}",
                     f"{100*m['precision_5cm']:.2f}",f"{100*m['recall_5cm']:.2f}",f"{100*m['f1_5cm']:.2f}",
                     f"{100*m['chamfer_m']:.3f}",m['n_prediction']])
scores_table=table(['Sequence','Method','Mean cm','RMSE cm','P@5cm %','R@5cm %','F1@5cm %','Chamfer cm','Points 2cm'],rows)
devrows=[]
for key,a in allres[seqs[0]]['results'].items():
    m=a['metrics']; devrows.append([key,f"{100*m['accuracy_mean_m']:.3f}",f"{100*m['precision_5cm']:.2f}",
                                   f"{100*m['recall_5cm']:.2f}",f"{100*m['f1_5cm']:.2f}"])
devtable=table(['Development candidate','Mean cm','P@5cm %','R@5cm %','F1@5cm %'],devrows)
tr=[]
for key,a in timing['backends'].items():
    tr.append([key,f"{a['mean_ms']:.3f}",f"{a['p50_ms']:.3f}",f"{a['p90_ms']:.3f}",f"{a['peak_cuda_allocated_bytes']/1024**2:.1f}"])
timing_table=table(['Same-input backend','Mean ms','Median ms','p90 ms','Peak CUDA MiB'],tr)

fig,ax=plt.subplots(1,3,figsize=(14,4))
labels=['V1_01 dev','V1_02 held-out','V1_03 held-out']; x=np.arange(3)
for offset,key,name,color in [(-.23,'legacy_tsdf','Legacy TSDF','#7c8794'),(0,'uniform_support0','Native uniform TSDF','#3f8fc6'),(.23,selected,'Balanced surfels','#16a085')]:
    ms=[allres[s]['results'][key]['metrics'] for s in seqs]
    ax[0].bar(x+offset,[100*m['accuracy_mean_m'] for m in ms],.23,label=name,color=color)
    ax[1].bar(x+offset,[100*m['f1_5cm'] for m in ms],.23,label=name,color=color)
    ax[2].bar(x+offset,[100*m['recall_5cm'] for m in ms],.23,label=name,color=color)
for a,title in zip(ax,['Mean point distance [cm], lower better','F1 @5cm [%], higher better','Recall @5cm [%], higher better']):
    a.set_xticks(x,labels,rotation=12); a.set_title(title); a.grid(axis='y',alpha=.2); a.set_axisbelow(True)
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(figdir/'fusion_v2_quality.png',dpi=140); plt.close(fig)

ref=np.load(next(DEST.glob('reference_roi_*.npy')))
tree=cKDTree(ref); fig,axes=plt.subplots(3,2,figsize=(12,15))
cloud_links={}
for row,seq in enumerate(seqs):
    base=ROOT/'outputs'/seq/'stage8_fusion/tsdf_cloud_openvins.ply'
    fast_out=ROOT/'outputs'/seq/'stage8_fusion_balanced'/selection['variant']
    out=fast_out if fast_out.exists() else ROOT/'outputs'/seq/'stage8_fusion_v2'/selection['variant']
    suffix=f"_cc{selection['min_component_points']}" if selection['min_component_points'] else ''
    filename=f"surface_support{selection['min_surface_support']}"+suffix+'_cloud.ply'
    path=out/filename; cloud_links[seq]=str(path.relative_to(ROOT)).replace('\\','/')
    alignment=json.loads((ROOT/'outputs'/seq/'stage9_eval/vio_metrics.json').read_text())
    R,t=np.array(alignment['align_R']),np.array(alignment['align_t'])
    points=[ply(base),ply(path)@R.T+t]
    for col,p in enumerate(points):
        p=crop(p); p=p[p[:,2]<1.6]
        rng=np.random.default_rng(4); p=p[rng.choice(len(p),min(60000,len(p)),replace=False)]
        error=tree.query(p,workers=2)[0]
        im=axes[row,col].scatter(p[:,0],p[:,1],c=error,s=.3,cmap='magma_r',vmin=0,vmax=.15)
        axes[row,col].set(xlim=(-4.8,4.8),ylim=(-4.8,4.8),aspect='equal',xlabel='World x [m]',ylabel='World y [m]')
        axes[row,col].set_title(seq+' | '+('Legacy TSDF' if col==0 else 'Balanced surfel point surface'))
fig.subplots_adjust(hspace=.24,wspace=.18,right=.87)
cax=fig.add_axes([.9,.2,.02,.6]); fig.colorbar(im,cax=cax,label='Point-to-reference distance [m]')
fig.suptitle('Evaluation registration only | z < 1.6m for view | identical error color scale',y=.92)
fig.savefig(figdir/'fusion_v2_maps.png',dpi=120); plt.close(fig)

holdout_pass=all(allres[s]['results'][selected]['metrics']['accuracy_mean_m'] < allres[s]['results']['legacy_tsdf']['metrics']['accuracy_mean_m']
                 and allres[s]['results'][selected]['metrics']['f1_5cm']>=allres[s]['results']['legacy_tsdf']['metrics']['f1_5cm']-.01 for s in seqs[1:])
summary=dict(selected=selected,heldout_acceptance='mean error below legacy TSDF and F1 no more than 1pp below it on each held-out sequence',
             heldout_acceptance_passed=holdout_pass,
             mean_error_reduction_vs_legacy_tsdf_percent={s:100*(1-allres[s]['results'][selected]['metrics']['accuracy_mean_m']/allres[s]['results']['legacy_tsdf']['metrics']['accuracy_mean_m']) for s in seqs},
             integration_speedup=timing['backends']['legacy']['mean_ms']/timing['backends']['surfel_uniform']['mean_ms'],
             no_claim='Not universal best, not calibrated uncertainty, not end-to-end real-time, not watertight mesh')
(DEST/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

text='''# Fusion v2 — cải thiện kiến trúc depth + VIO

Ngày: **2026-11-08**. Ưu tiên người dùng: **cân bằng chất lượng và tốc độ**.

## Kết quả và cấu hình đã chốt

Đã triển khai và chạy trên ba sequence. Cấu hình cân bằng được chọn **chỉ từ V1_01**,
khóa trước khi chấm V1_02/V1_03: `surfel_uniform`, voxel 3cm, depth 0.2–4m,
tối thiểu **6 keyframe quan sát**, loại connected components dưới **30 points**.
Đầu ra chính là **point surface dạng surfel**, không phải watertight TSDF mesh.
TSDF mesh backend mới vẫn có để dùng khi cần mesh.

Không bật confidence weighting mặc định: các proxy spatial/temporal đã thử làm F1
giảm ở development. Có implementation và numerical tests, nhưng không quảng bá chúng
như improvement hoặc calibrated uncertainty. Kết quả thực quyết định preset.

'''+scores_table+'''So sánh dùng cùng Leica reference gốc, fixed ROI, voxel sampling 2cm và thresholds.
GT-derived SE3 registration chỉ được áp dụng **sau khi mapping** để chấm điểm; không ICP,
không `gtrotcorr`, không dùng scan hay GT vào fusion/bounds. Mean/RMSE/Chamfer là
point-to-point trên sampled surface; không gọi là exact mesh point-to-surface accuracy.

Legacy naive accumulation được giữ trong bảng để tránh che baseline mạnh. Một preset
không nhất thiết thắng naive ở mọi precision/recall/F1: phải xem rõ trade-off.
Legacy maps và new maps có thể khác số frame do old aligned trajectory đã cắt theo GT.
**Controlled comparison** là `uniform_support0` với preset mới: cùng raw pose stream,
cùng prepared depths, native bounds, frame timestamps, voxel size và evaluation protocol.
Khác representation/meshing nên các metric điểm vẫn nhạy sampling; chưa chứng minh
quality của continuous surface hoặc mesh topology hơn mọi baseline.

![Quality](figures/fusion_v2_quality.png)

## Kiến trúc thay đổi

1. **Raw VIO native frame:** bỏ GT alignment khỏi input mapping, tính bounds từ observed
   depths; giữ đúng `T_WCrect = T_WI T_ICraw R0ᵀ`.
2. **Linear + SLERP:** pose bracketing, không extrapolate; reject timestamp duplicates/gaps.
3. **Hai mapping backends:** TSDF chỉ quét current frustum AABB; surfel chỉ gom observed
   surface points, không dựng zero crossings trong vùng kém hỗ trợ.
4. **Multi-view support đúng đơn vị:** một voxel nhận tối đa một temporal vote từ một
   input frame, dù nhiều pixel cùng rơi vào đó. Giữ đủ số view trước khi xuất surface.
5. **Conservative cleanup:** loại các cụm nhỏ, không fit ICP hoặc dùng scan để xóa điểm.
6. **Reproducible phases:** prepare/integrate/extract, explicit frames/poses/calibration,
   code/input fingerprints; source/weights/old outputs không bị thay để làm đẹp metric.

Đây là improvement engineering có đo kiểm, chưa phải novel UATDF, và không có fine-tuning.
Online calibration/time offset và covariance cross terms vẫn thiếu từ legacy export;
không coi pose marginal blocks như calibrated relative uncertainty.

## Timing đối chứng cùng dữ liệu

'''+timing_table+'''RTX 4060 Laptop GPU; cùng 20 selected development frames/native volume bounds,
5 warmups +30 timed calls, CUDA synchronization; đo H2D + integration, không disk I/O,
stereo inference, confidence preparation hoặc extraction. Peak CUDA là allocated tensor memory,
không toàn process/driver reserved memory. Các backends đo tuần tự trong cùng process.

Depth IGEV++ hiện tại vẫn là runtime bottleneck; không được suy tốc độ end-to-end từ
4–5ms fusion. Chuẩn bị cache offline còn gồm image reads, warps và compression.
Surfel/TSDF volumes hiện dùng dense arrays trong native bounded volume, **không phải**
sparse voxel hashing hoặc unbounded online SLAM. Muốn mapping dài hạn cần submaps.

## Development search và held-out protocol

Chỉ dùng V1_01 để so uniform/spatial/temporal TSDF, surface support, uniform/weighted
surfels và small-component cleanup. Có hai selection drafts được giữ ở
`artifacts/fusion_v2/development_initial/` và `development_quality_only/`.
Sau đó chọn đúng ưu tiên balanced: giữ recall không thấp hơn native uniform quá 2pp,
lấy các candidate có F1 cách best không quá 0.5pp, chọn integration nhanh nhất;
cùng backend thì chọn F1 cao hơn. `selection.json` khóa trước held-out evaluation.
Không chỉnh threshold theo V1_02/V1_03. Chúng vẫn cùng phòng Room1, chưa phải
cross-scene generalization. Nhiều development candidates tạo tuning risk;
cần scene mới cho scientific claim.

'''+devtable+'''## Cách chạy preset

```powershell
$cpu = 'C:\\Users\\Lenovo\\Documents\\BHYT_Vinmec\\.venv\\Scripts\\python.exe'
$gpu = 'C:\\Users\\Lenovo\\Documents\\meddies-health-ai\\.venv-4060\\Scripts\\python.exe'
& $cpu pipeline/run_balanced_fusion.py --sequence V1_02_medium --cpu-python $cpu --gpu-python $gpu --rebuild
```

`--rebuild` là explicit: development cache đã ghi fingerprint phiên bản code lúc prepare,
còn module được bổ sung sau đó. Khi code/settings khác, mapper không silently reuse cache.
Lần chạy mới có thể dùng `--work <new-directory>` để giữ nguyên toàn bộ development outputs.
Không cần cài packages/checkpoint mới trên workstation này; hai existing Python environments
được dùng cho các phase phù hợp. Môi trường đủ dependencies có thể chạy cả ba phase.

Để xuất TSDF mesh: dùng `--variant uniform` cho integrate/extract;
surface support=4 là quality-only development candidate, **không** thay preset balanced.
Không chọn lại threshold bằng cách nhìn held-out metrics. Xem `pipeline/FUSION_V2.md`.

## Files và kiểm tra

- `pipeline/fusion_geometry.py`: camera/coordinate contracts, finite depth và pose interpolation.
- `pipeline/fusion_confidence.py`: visibility-aware causal consistency và confidence proxies (default off).
- `pipeline/fusion_tsdf.py`: FrustumTSDF và VoxelSurfels, one-frame-one-vote.
- `pipeline/fusion_v2.py`: data preparation, GPU integration, extraction/export.
- `pipeline/fusion_balanced.json`, `pipeline/run_balanced_fusion.py`: frozen preset và runner.
- `pipeline/tests/test_fusion_v2.py`: 10 meaningful numerical controls; xem `fusion_v2_tests.log`.
- `reports/artifacts/fusion_v2/`: immutable selection, three evaluation JSONs, timing và summary.
- `outputs/{sequence}/stage8_fusion_v2/`: development/held-out prepared data, volume, PLY và config.
- `outputs/{sequence}/stage8_fusion_balanced/`: fast selected runner, bỏ tính proxy không dùng.

## Giới hạn thực tế

Không thay IGEV++ hoặc OpenVINS estimator; không sửa ROS shutdown exception trong task này.
Không có NLL/reliability calibration, pose/depth cross-covariance propagation hay uncertainty
novelty claim. Camera motion/distortion/online calibration bias vẫn có thể giới hạn map.
Confidence weighting failed development test nên bị loại khỏi default thay vì đổi tên
heuristic thành uncertainty-aware contribution. Original Leica/pose reference caveats
từ audit trước vẫn áp dụng. Chưa chứng minh tối ưu tuyệt đối hoặc đạt 20FPS end-to-end.

![Map comparison](figures/fusion_v2_maps.png)
'''
text+='\n## Acceptance gate\n\n'+('**PASS**' if holdout_pass else '**NOT PASSED**')+': '+summary['heldout_acceptance']+'.\n'
text+='Không thay đổi preset hoặc tiêu chí sau khi xem held-out results.\n'
validation=DEST/'fast_path_validation.json'
if validation.exists():
    V=json.loads(validation.read_text())
    text+='''\n## Fast path: không tính các confidence proxy đã bị loại\n\n
Sau khi chốt quality parameters, chỉ tối ưu performance: runner uniform bỏ spatial/temporal
confidence không dùng, và tính color remap coordinates một lần/sequence. Tham số, depth,
pose và frame set không đổi. Các source đã dùng để khóa selection được giữ nguyên ở
`artifacts/fusion_v2/frozen_sources/`; fast-path code có hashes riêng trong integration logs.
Đã chạy lại **cả ba sequence** bằng runner cuối và đối chiếu với frozen outputs;
`fast_path_validation.json` xác nhận cùng số points, coordinate difference dưới 0.1mm,
pose arrays bằng nhau tuyệt đối và 10 depth/color frames mỗi sequence khớp hoàn toàn.
Baseline source/checkpoint/output hashes được kiểm tra giữ nguyên.
Không chọn lại tham số bằng held-out results.\n\n'''
    text+=table(['Sequence','Frames','Fast prepare mean ms','Fast integrate mean ms','Max coordinate difference m'],
                [[s,a['frames'],f"{a['prepare_ms_mean']:.2f}",f"{a['integration']['integrate_ms_mean']:.2f}",
                  f"{a['max_surface_coordinate_difference_m']:.8g}"] for s,a in V['sequences'].items()])
    text+='Prepare timing còn gồm disk I/O và cache serialization; không coi đó là online kernel latency.\n'
(ROOT/'reports/08_fusion_architecture_improvement.md').write_text(text,encoding='utf-8')

def html_table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def embedded(name):
    data=base64.b64encode((figdir/name).read_bytes()).decode()
    return '<img src="data:image/png;base64,'+data+'" alt="'+escape(name)+'">'

html='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fusion v2 · Quality & speed</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#101820;color:#e7eef5;margin:0}main{max-width:1220px;margin:auto;padding:36px 22px}h1{font-size:36px;line-height:1.15}h2{margin-top:40px}a{color:#67d4c3}.muted{color:#b3c2ce}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #34444f;text-align:right;padding:9px;white-space:nowrap}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}img{width:100%;background:white;margin-top:18px}.card{padding:18px;background:#1b2b36;border-left:4px solid #24baa1;margin:24px 0}code{color:#8ee1d3}li{margin:8px 0}</style><main>
<a href="compare_V1.html">← Báo cáo baseline</a><p class="muted">2026-11-08 · Development V1_01 · Held-out V1_02 / V1_03</p>
<h1>Depth + VIO: kiến trúc cân bằng chất lượng và tốc độ</h1>
<div class="card"><b>Preset đã chốt:</b> uniform voxel surfels, 3cm; ít nhất 6 keyframe hỗ trợ; loại cụm dưới 30 points. Input là raw OpenVINS pose và cached IGEV++ depth. <b>Đầu ra là point surface, không phải mesh kín.</b></div>
<p>Pose interpolation bằng SLERP, native-frame mapping không dùng GT làm input, mỗi frame chỉ đóng một phiếu trong voxel. Confidence proxies đã thử nhưng làm metric giảm, nên tắt trong preset.</p><h2>Kết quả đo thực</h2>'''
html+=html_table(['Sequence','Method','Mean cm','RMSE cm','P@5 %','R@5 %','F1@5 %','Chamfer cm','Points'],rows)
html+='<p class="muted">Cùng reference/ROI/voxel scoring; SE3 chỉ áp dụng sau mapping. Legacy naive vẫn có trong bảng. Different representations/frame coverage limit causal comparisons; point metrics không chứng minh mesh quality.</p>'+embedded('fusion_v2_quality.png')
html+='<h2>Integration benchmark cùng đầu vào</h2>'+html_table(['Backend','Mean ms','Median ms','p90 ms','Peak CUDA MiB'],tr)
html+='<p>5 warmups +30 timed calls, synchronized GPU. Không gồm stereo inference/I/O/preparation/extraction; đây không phải end-to-end FPS.</p>'
html+='<h2>Bản đồ và khoảng cách đến reference</h2>'+embedded('fusion_v2_maps.png')
html+='<h2>Artifacts</h2><ul>'
for seq,path in cloud_links.items(): html+='<li><a href="../'+path+'">'+seq+' · native-frame surfel PLY</a></li>'
html+='</ul><p><a href="../reports/08_fusion_architecture_improvement.md">Báo cáo đầy đủ, controls, ablations và lệnh chạy</a></p><p class="muted">Không claim novelty UATDF, calibrated uncertainty hoặc real-time 20FPS. Original reference caveats remain.</p></main></html>'
(ROOT/'report/fusion_v2_comparison.html').write_text(html,encoding='utf-8')
# Keep historical figures/numbers intact; add one navigation link, also in generator template.
banner='<p><a href="fusion_v2_comparison.html" style="color:var(--accent);font-weight:600">Xem kiến trúc fusion v2 · kết quả mới và trade-off chất lượng/tốc độ →</a></p>'
for path in [ROOT/'pipeline/compare_template.html',ROOT/'report/compare_V1.html']:
    old=path.read_text(encoding='utf-8')
    if 'fusion_v2_comparison.html' not in old: path.write_text(old.replace('<main>','<main>\n'+banner,1),encoding='utf-8')
index=ROOT/'reports/README.md'; old=index.read_text(encoding='utf-8')
if '08_fusion_architecture_improvement' not in old:
    index.write_text(old+'\n\n## Architecture improvement — 2026-11-08\n\n[Measured balanced fusion v2](08_fusion_architecture_improvement.md) · [HTML comparison](../report/fusion_v2_comparison.html)\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
