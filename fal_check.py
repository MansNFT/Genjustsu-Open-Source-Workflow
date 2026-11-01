"""Tiny probe for the fal SAM 3 RLE response format.

fal documents `fal-ai/sam-3/video-rle` as returning per-frame RLE masks but does not
state the encoding. Run this once on a short clip to confirm the response decodes
before processing real jobs:

    .venv/bin/python fal_check.py short-clip.mp4 [--prompt person] [--max-frames 200]

It uploads the clip, runs SAM 3 once (PAID, billed per 16 frames), saves the raw response
and reports whether it decodes into one mask per frame. It also compares each decoded mask
with the bounding box fal returns for the same frame, which catches a wrong pixel order.

To decode a response you already paid for, without a new request:

    .venv/bin/python fal_check.py short-clip.mp4 --reuse
"""
import argparse,json,sys
from config import ROOT

def describe(value,depth=0):
    if isinstance(value,dict):return {k:describe(v,depth+1) if depth<2 else type(v).__name__ for k,v in list(value.items())[:12]}
    if isinstance(value,list):return f'list[{len(value)}] of '+(type(value[0]).__name__ if value else 'empty')+(' · first: '+repr(value[0])[:160] if value else '')
    if isinstance(value,str):return f'str[{len(value)}]: '+value[:80]
    return value

def box_overlap(masks,boxes):
    """Overlap (IoU) between each mask's outline and fal's normalized [cx,cy,w,h] box."""
    import numpy as np
    scores=[]
    for mask,box in zip(masks,boxes or []):
        if not box or len(box)!=4 or not mask.any():continue
        ys,xs=np.where(mask);hh,ww=mask.shape
        mine=(xs.min()/ww,ys.min()/hh,(xs.max()+1)/ww,(ys.max()+1)/hh)
        cx,cy,bw,bh=box;theirs=(cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2)
        ix=max(0,min(mine[2],theirs[2])-max(mine[0],theirs[0]));iy=max(0,min(mine[3],theirs[3])-max(mine[1],theirs[1]))
        inter=ix*iy;union=(mine[2]-mine[0])*(mine[3]-mine[1])+(theirs[2]-theirs[0])*(theirs[3]-theirs[1])-inter
        if union>0:scores.append(inter/union)
    return scores

def main():
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--prompt',default='person');p.add_argument('--max-frames',type=int,default=96)
    p.add_argument('--reuse',action='store_true',help='decode the saved response instead of sending a new paid request')
    a=p.parse_args()
    from raw_masks import video_info,SAM_ENDPOINT
    from rle_masks import masks_from_response
    import requests
    frames,h,w=video_info(a.video)
    folder=ROOT/'runs'/'_fal_check';folder.mkdir(parents=True,exist_ok=True)
    saved=folder/'sam-check-result.json'
    if a.reuse:
        if not saved.exists():print('No saved response found at',saved);return 2
        result=json.loads(saved.read_text());print('Decoding the saved response; no new request is sent.')
    else:
        if frames>a.max_frames:print(f'Clip has {frames} frames; use a clip of at most {a.max_frames} frames, or raise the limit with --max-frames (cost is about $0.005 per 16 frames).');return 2
        from fal_provider import upload,run
        print(f'Uploading {frames} frames ({w}x{h}) and running {SAM_ENDPOINT} once...')
        result,request_id=run(SAM_ENDPOINT,{'video_url':upload(a.video),'prompt':a.prompt,'apply_mask':False,'detection_threshold':0.5},folder,'sam-check',print)
        print('request_id:',request_id)
    print('response shape:',json.dumps(describe(result),indent=2,default=str))
    def fetch(url):
        r=requests.get(url,timeout=180);r.raise_for_status();return r.json()
    try:
        masks=masks_from_response(result,frames,h,w,fetch)
    except Exception as exc:
        print('\nDECODE FAILED:',exc);print('Raw response saved in',saved,'- send me that file (it contains no credentials).');return 1
    areas=[int(m.sum()) for m in masks]
    print(f'\nDECODE OK: {len(masks)} masks of {masks[0].shape[1]}x{masks[0].shape[0]}; subject pixels per frame min/max = {min(areas)}/{max(areas)}')
    overlap=box_overlap(masks,result.get('boxes'))
    if overlap:
        print(f'Mask outline vs fal’s own box: average overlap {sum(overlap)/len(overlap):.0%}, worst frame {min(overlap):.0%}')
        if sum(overlap)/len(overlap)<0.8:print('WARNING: low overlap. The pixel order or indexing may be wrong; send me the saved response file and do not use this for real jobs yet.')
    else:print('No bounding boxes in the response, so the mask position could not be cross-checked.')
    if min(areas)==0:print('Some frames have an empty mask - check the prompt or detection threshold.')
    return 0

if __name__=='__main__':sys.exit(main())
