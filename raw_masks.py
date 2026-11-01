from pathlib import Path
import json
import av
import requests

SAM_ENDPOINT='fal-ai/sam-3/video-rle'

def video_info(path):
    """Frame count and size, counted by decoding so it matches what composite() sees."""
    with av.open(str(path)) as c:
        count=0;h=w=0
        for f in c.decode(video=0):
            if not count:h,w=f.height,f.width
            count+=1
    if not count:raise ValueError('Video has no frames.')
    return count,h,w

def segment_raw(video,folder,prompt='person',status=print):
    """Segment ``prompt`` with SAM 3 on fal and write one ``mask_NNNNN.png`` per frame (white = subject)."""
    import numpy as np
    from PIL import Image
    from fal_provider import upload,run
    from rle_masks import masks_from_response
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    frames,h,w=video_info(video)
    status('SAM 3 · uploading video to fal');url=upload(video)
    # apply_mask=False: only the per-frame RLE masks are needed, not a rendered overlay.
    result,_=run(SAM_ENDPOINT,{'video_url':url,'prompt':prompt,'apply_mask':False,'detection_threshold':0.5},folder,'sam-raw',status)
    def fetch(u):
        r=requests.get(u,timeout=180);r.raise_for_status();return r.json()
    masks=masks_from_response(result,frames,h,w,fetch)
    out=folder/'raw-masks';out.mkdir(exist_ok=True)
    for i,m in enumerate(masks):Image.fromarray(m.astype(np.uint8)*255).save(out/f'mask_{i:05d}.png')
    return out

def clean_border_fragments(folder):
    """Remove tiny, narrow disconnected edge fragments; preserve large edge subjects."""
    import numpy as np
    from PIL import Image
    from scipy.ndimage import label
    folder=Path(folder);output=folder.parent/'clean-masks';output.mkdir(exist_ok=True)
    report=[]
    for p in sorted(folder.glob('mask_*.png')):
        a=np.asarray(Image.open(p).convert('L'))>127;labels,_=label(a);h,w=a.shape;removed=0
        for ident in (set(labels[:,0])|set(labels[:,-1]))-{0}:
            y,x=np.where(labels==ident)
            if len(x)<h*w*.03 and x.max()-x.min()+1<=max(6,int(w*.05)):
                a[labels==ident]=False;removed+=len(x)
        Image.fromarray(a.astype(np.uint8)*255).save(output/p.name)
        report.append({'frame':p.stem,'removed_border_pixels':removed,'mask_pixels':int(a.sum()),'left_right_edge_pixels':int(a[:,0].sum()+a[:,-1].sum())})
    (folder.parent/'mask-cleanup-report.json').write_text(json.dumps(report,indent=2))
    return output
