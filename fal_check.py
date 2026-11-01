"""Tiny PAID probe for the one fal response whose format is not documented.

fal documents `fal-ai/sam-3/video-rle` as returning per-frame RLE masks but does not
state the encoding. Run this once on a very short clip (1-2 seconds) to confirm the
response decodes before processing real jobs:

    .venv/bin/python fal_check.py short-clip.mp4 [--prompt person]

It uploads the clip, runs SAM 3 once (billed per 16 frames), saves the raw response and
reports whether it decodes into one mask per frame. Nothing else is submitted.
"""
import argparse,json,sys
from config import ROOT

def describe(value,depth=0):
    if isinstance(value,dict):return {k:describe(v,depth+1) if depth<2 else type(v).__name__ for k,v in list(value.items())[:12]}
    if isinstance(value,list):return f'list[{len(value)}] of '+(type(value[0]).__name__ if value else 'empty')+(' · first: '+repr(value[0])[:160] if value else '')
    if isinstance(value,str):return f'str[{len(value)}]: '+value[:80]
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--prompt',default='person');p.add_argument('--max-frames',type=int,default=96)
    a=p.parse_args()
    from raw_masks import video_info,SAM_ENDPOINT
    from fal_provider import upload,run
    from rle_masks import masks_from_response
    import requests
    frames,h,w=video_info(a.video)
    if frames>a.max_frames:print(f'Clip has {frames} frames; use a clip of at most {a.max_frames} frames (about 4 s at 24 fps) for this probe.');return 2
    folder=ROOT/'runs'/'_fal_check';folder.mkdir(parents=True,exist_ok=True)
    print(f'Uploading {frames} frames ({w}x{h}) and running {SAM_ENDPOINT} once...')
    result,request_id=run(SAM_ENDPOINT,{'video_url':upload(a.video),'prompt':a.prompt,'apply_mask':False,'detection_threshold':0.5},folder,'sam-check',print)
    print('request_id:',request_id);print('response shape:',json.dumps(describe(result),indent=2,default=str))
    def fetch(url):
        r=requests.get(url,timeout=180);r.raise_for_status();return r.json()
    try:
        masks=masks_from_response(result,frames,h,w,fetch)
    except Exception as exc:
        print('\nDECODE FAILED:',exc);print('Raw response saved in',folder/'sam-check-result.json','- send me that file (it contains no credentials).');return 1
    areas=[int(m.sum()) for m in masks]
    print(f'\nDECODE OK: {len(masks)} masks of {masks[0].shape[1]}x{masks[0].shape[0]}; subject pixels per frame min/max = {min(areas)}/{max(areas)}')
    if min(areas)==0:print('Some frames have an empty mask - check the prompt or detection threshold.')
    return 0

if __name__=='__main__':sys.exit(main())
