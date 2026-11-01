"""Decode per-frame run-length-encoded masks (COCO conventions) into boolean arrays.

fal's ``fal-ai/sam-3/video-rle`` documents a per-frame ``rle`` list but not the exact
encoding, so this accepts every common shape and fails loudly on anything else:

* dict ``{"size":[h,w],"counts":<str|list>}``
* a JSON string of that dict
* a bare compressed counts string or an uncompressed counts list (size taken from the video)
* a list of the above for several objects in one frame (merged with OR)
"""
import json
import numpy as np

def counts_from_string(s):
    """COCO compressed counts string -> list of run lengths (mirrors pycocotools rleFrString)."""
    if isinstance(s,(bytes,bytearray)):s=s.decode('ascii')
    counts=[];p=0
    while p<len(s):
        x=0;k=0;more=True
        while more:
            c=ord(s[p])-48
            x|=(c&0x1f)<<(5*k);more=bool(c&0x20);p+=1;k+=1
            if not more and (c&0x10):x|=-1<<(5*k)
        if len(counts)>2:x+=counts[-2]
        counts.append(x)
    return counts

def counts_to_mask(counts,h,w):
    flat=np.zeros(h*w,dtype=bool);pos=0;value=False
    for run in counts:
        if run<0:raise ValueError('Negative run length in RLE mask.')
        if value:flat[pos:pos+run]=True
        pos+=run;value=not value
    if pos!=h*w:raise ValueError(f'RLE covers {pos} pixels but a {h}x{w} frame has {h*w}.')
    return flat.reshape((h,w),order='F')  # COCO RLE is column-major

def decode_one(item,h,w):
    """One RLE item -> bool mask. ``h``,``w`` are used when the item carries no size."""
    if item is None or item=='' or item==[] or item=={}:return np.zeros((h,w),dtype=bool)
    if isinstance(item,(bytes,bytearray)):item=item.decode('ascii')
    if isinstance(item,str) and item.lstrip().startswith(('{','[')):item=json.loads(item)
    if isinstance(item,dict):
        counts=item.get('counts')
        size=item.get('size') or [h,w]
        if counts is None:raise ValueError('RLE dict has no "counts".')
        h,w=int(size[0]),int(size[1])
    else:counts=item
    if isinstance(counts,(bytes,bytearray)):counts=counts.decode('ascii')
    if isinstance(counts,str):counts=counts_from_string(counts)
    if not isinstance(counts,(list,tuple)):raise ValueError(f'Unrecognised RLE item of type {type(item).__name__}.')
    return counts_to_mask(counts,h,w)

def decode_frame(item,h,w):
    """A frame entry may hold one RLE or a list of RLEs (several objects): merge with OR."""
    if isinstance(item,list) and item and all(isinstance(x,(dict,str,list)) for x in item) and not all(isinstance(x,int) for x in item):
        masks=[decode_one(x,h,w) for x in item]
        if len({m.shape for m in masks})!=1:raise ValueError('Objects in one frame have different mask sizes.')
        return np.logical_or.reduce(masks)
    return decode_one(item,h,w)

def masks_from_response(result,frames,h,w,fetch=None):
    """fal result dict -> list of ``frames`` boolean masks.

    ``fetch(url)`` must return parsed JSON; it is used when ``rle`` is delivered as a file.
    """
    rle=result.get('rle')
    if isinstance(rle,dict) and rle.get('url'):
        if fetch is None:raise ValueError('rle is a file URL but no fetch function was given.')
        rle=fetch(rle['url'])
    if isinstance(rle,dict) and 'rle' in rle:rle=rle['rle']
    if not isinstance(rle,list):raise ValueError(f'Unexpected "rle" type {type(rle).__name__}.')
    if len(rle)!=frames:
        raise ValueError(f'SAM returned {len(rle)} RLE entries for {frames} frames; refusing to guess how objects and frames are ordered. Run fal_check.py and inspect the response.')
    masks=[decode_frame(item,h,w) for item in rle]
    if len({m.shape for m in masks})!=1:raise ValueError('Frames returned masks of different sizes.')
    return masks
