import unittest,tempfile,json,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import rle_masks,fal_provider

def mask_counts(mask):
    f=mask.ravel(order='F').astype(np.int8)
    bounds=np.concatenate([[0],np.flatnonzero(np.diff(f))+1,[f.size]])
    runs=np.diff(bounds).tolist()
    return ([0] if f[0]==1 else [])+runs

def counts_to_string(counts):
    """COCO rleToString, written independently of the decoder under test."""
    out=[]
    for i,x in enumerate(counts):
        x=int(x)
        if i>2:x-=counts[i-2]
        more=True
        while more:
            c=x&0x1f;x>>=5
            more=(x!=-1) if (c&0x10) else (x!=0)
            if more:c|=0x20
            out.append(chr(c+48))
    return ''.join(out)

def sample(h=24,w=40,seed=0):
    rng=np.random.default_rng(seed);m=np.zeros((h,w),dtype=bool)
    m[4:15,8:30]=True;m[rng.integers(0,h),rng.integers(0,w)]=True;m[0,0]=True
    return m

class RleTests(unittest.TestCase):
    def test_compressed_string_roundtrip(self):
        for seed in range(5):
            m=sample(seed=seed);s=counts_to_string(mask_counts(m))
            self.assertTrue((rle_masks.decode_one(s,*m.shape)==m).all())
    def test_dict_json_and_uncompressed_forms(self):
        m=sample();counts=mask_counts(m);s=counts_to_string(counts);h,w=m.shape
        for item in [{'size':[h,w],'counts':s},json.dumps({'size':[h,w],'counts':s}),counts,{'size':[h,w],'counts':counts}]:
            self.assertTrue((rle_masks.decode_one(item,h,w)==m).all(),item)
    def test_mask_starting_with_foreground(self):
        m=np.zeros((6,7),dtype=bool);m[:,0]=True
        self.assertTrue((rle_masks.decode_one(counts_to_string(mask_counts(m)),6,7)==m).all())
    def test_dict_size_overrides_video_size(self):
        m=sample(12,20);d={'size':list(m.shape),'counts':counts_to_string(mask_counts(m))}
        self.assertEqual(rle_masks.decode_one(d,99,99).shape,(12,20))
    def test_multiple_objects_are_merged(self):
        a=np.zeros((10,10),dtype=bool);a[:5,:5]=True;b=np.zeros((10,10),dtype=bool);b[5:,5:]=True
        item=[counts_to_string(mask_counts(a)),counts_to_string(mask_counts(b))]
        self.assertTrue((rle_masks.decode_frame(item,10,10)==(a|b)).all())
    def test_empty_frame_is_empty_mask(self):
        for empty in [None,'',[],{}]:self.assertFalse(rle_masks.decode_one(empty,5,6).any())
    def test_wrong_size_is_rejected(self):
        s=counts_to_string(mask_counts(sample()))
        with self.assertRaises(ValueError):rle_masks.decode_one(s,10,10)
    def test_frame_count_mismatch_is_rejected(self):
        m=sample();s=counts_to_string(mask_counts(m))
        with self.assertRaisesRegex(ValueError,'refusing to guess'):rle_masks.masks_from_response({'rle':[s,s,s]},2,*m.shape)
    def test_file_mode_uses_fetch(self):
        m=sample();s=counts_to_string(mask_counts(m))
        got=rle_masks.masks_from_response({'rle':{'url':'https://x/y.json'}},2,*m.shape,fetch=lambda u:{'rle':[s,s]})
        self.assertEqual(len(got),2);self.assertTrue((got[0]==m).all())
    def test_matches_pycocotools_when_available(self):
        try:from pycocotools import mask as coco
        except ImportError:self.skipTest('pycocotools not installed')
        m=sample(31,47,seed=3);enc=coco.encode(np.asfortranarray(m.astype(np.uint8)))
        self.assertEqual(counts_to_string(mask_counts(m)),enc['counts'].decode())
        self.assertTrue((rle_masks.decode_one({'size':list(m.shape),'counts':enc['counts']},*m.shape)==m).all())

class FakeHandler:
    request_id='req-123'
    def __init__(self,result,folder=None):self.result=result;self.folder=folder
    def iter_events(self,with_logs=False,interval=0.1):
        # The request ID file must exist before any waiting happens.
        if self.folder:assert (Path(self.folder)/'t-request.json').exists()
        return iter([])
    def get(self):return self.result
class FakeClient:
    def __init__(self,result,folder=None):self.result=result;self.folder=folder;self.calls=[]
    def submit(self,endpoint,arguments):self.calls.append((endpoint,arguments));return FakeHandler(self.result,self.folder)
    def upload_file(self,path):return 'https://fal.media/files/in.mp4'

class ProviderTests(unittest.TestCase):
    def test_key_required_and_placeholder_rejected(self):
        with patch.dict(os.environ,{'FAL_KEY':'','FAL_API_KEY':''}),patch('fal_provider.setting',lambda n,d='':''):
            with self.assertRaisesRegex(ValueError,'FAL_KEY'):fal_provider.api_key()
        with patch('fal_provider.setting',lambda n,d='':'your_key_here' if n=='FAL_KEY' else ''):
            with self.assertRaises(ValueError):fal_provider.api_key()
    def test_run_saves_request_id_before_waiting_and_returns_result(self):
        with tempfile.TemporaryDirectory() as d:
            fake=FakeClient({'ok':True},d)
            with patch('fal_provider.client',return_value=fake):
                result,rid=fal_provider.run('fal-ai/x',{'a':1},d,'t')
            self.assertEqual((result,rid),({'ok':True},'req-123'))
            saved=json.loads((Path(d)/'t-request.json').read_text())
            self.assertEqual(saved['request_id'],'req-123');self.assertNotIn('key',json.dumps(saved).lower())
    def test_file_url(self):
        self.assertEqual(fal_provider.file_url({'url':'https://a/b'}),'https://a/b')
        with self.assertRaises(ValueError):fal_provider.file_url({})

class StageTests(unittest.TestCase):
    def test_segment_raw_writes_composite_compatible_masks(self):
        import pipeline,raw_masks
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);video=d/'v.mp4';out,s=pipeline.writer(video,32,16)
            for i in range(12):pipeline.put(out,s,np.full((16,32,3),i*10,dtype=np.uint8),i)
            pipeline.finish(out,s)
            m=np.zeros((16,32),dtype=bool);m[4:12,8:20]=True;rle=counts_to_string(mask_counts(m))
            with patch('fal_provider.upload',return_value='https://fal.media/in.mp4'),patch('fal_provider.run',return_value=({'rle':[rle]*12},'req-1')) as run:
                masks=raw_masks.segment_raw(video,d/'job','person',lambda s:None)
            files=sorted(masks.glob('mask_*.png'))
            self.assertEqual([int(p.stem.split('_')[-1]) for p in files],list(range(12)))
            self.assertTrue((np.asarray(Image.open(files[3]).convert('L'))>127).tolist()==m.tolist())
            args=run.call_args[0]
            self.assertEqual(args[0],'fal-ai/sam-3/video-rle');self.assertEqual(args[1]['prompt'],'person');self.assertFalse(args[1]['apply_mask'])
    def test_depth_fal_requests_pinned_colormap_and_downloads(self):
        import pipeline
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);src=d/'o.mp4';src.write_bytes(b'x')
            def fake_download(url,dest,timeout=0):Path(dest).write_bytes(b'depth');return dest
            with patch('fal_provider.upload',return_value='https://fal.media/in.mp4'),patch('fal_provider.run',return_value=({'video':{'url':'https://fal.media/d.mp4'}},'r')) as run,patch('fal_provider.download',fake_download),patch('config.setting',lambda n,default='':default):
                pipeline.depth_fal(src,d/'depth-raw.mp4',lambda s:None)
            endpoint,args=run.call_args[0][:2]
            self.assertEqual(endpoint,'fal-ai/depth-anything-video');self.assertEqual(args['colormap'],'inferno');self.assertEqual(args['model'],'VDA-Large')
            self.assertEqual((d/'depth-raw.mp4').read_bytes(),b'depth')
    def test_separate_pins_htdemucs_and_builds_manifest(self):
        import audio_workflow as aw
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);rate=aw.RATE;t=np.arange(rate)/rate
            tone=(0.3*np.sin(2*np.pi*440*t)).astype(np.float32);stereo=np.stack([tone,tone]);aw.write_audio(d/'src.wav',stereo)
            def fake_download(url,dest,timeout=0):aw.write_audio(dest,stereo*0.5);return dest
            result={n:{'url':f'https://fal.media/{n}.wav'} for n in ['vocals','drums','bass','other']}
            with patch('fal_provider.upload',return_value='https://fal.media/a.wav'),patch('fal_provider.run',return_value=(result,'req-9')) as run,patch('fal_provider.download',fake_download):
                aw.separate(d/'src.wav',d,lambda s:None)
            args=run.call_args[0]
            self.assertEqual(args[0],'fal-ai/demucs');self.assertEqual(args[1]['model'],'htdemucs');self.assertEqual(args[1]['output_format'],'wav')
            manifest=json.loads((d/'audio-manifest.json').read_text())
            self.assertEqual(manifest['request_id'],'req-9');self.assertTrue((d/'music.wav').exists())

if __name__=='__main__':unittest.main()
