"""fal.ai adapter: upload, queued runs with persisted request IDs, downloads.

Replaces the former Replicate calls. The module is deliberately not named
``fal_client`` so it never shadows the installed ``fal-client`` package.
"""
from pathlib import Path
import json,time
import requests
from config import setting

PLACEHOLDERS=['your_key','your_api','replace_me','placeholder']
DEFAULT_TIMEOUT=900

def api_key():
    value=(setting('FAL_KEY') or setting('FAL_API_KEY')).strip()
    if not value or any(x in value.lower() for x in PLACEHOLDERS):
        raise ValueError('FAL_KEY is required: set it in this project’s .env file (https://fal.ai/dashboard/keys).')
    return value

def client():
    import fal_client
    return fal_client.SyncClient(key=api_key())

def upload(path):
    """Upload a local file to fal storage and return its URL."""
    return client().upload_file(Path(path))

def file_url(value):
    """fal File objects are dicts with a ``url``; tolerate a bare URL string."""
    if isinstance(value,dict) and value.get('url'):return value['url']
    if isinstance(value,str) and value.startswith('http'):return value
    raise ValueError('fal did not return a downloadable file URL.')

def download(url,dest,timeout=180):
    r=requests.get(url,timeout=timeout);r.raise_for_status();Path(dest).write_bytes(r.content);return dest

def run(endpoint,arguments,folder,tag,status=None,timeout=DEFAULT_TIMEOUT):
    """Submit to the fal queue and wait. The request ID is saved before waiting.

    Returns ``(result, request_id)``. On a timeout the paid request keeps running on
    fal: use ``collect(endpoint, request_id)`` instead of submitting again.
    """
    folder=Path(folder);c=client();handler=c.submit(endpoint,arguments=arguments)
    request_id=handler.request_id
    (folder/f'{tag}-request.json').write_text(json.dumps({'provider':'fal','endpoint':endpoint,'request_id':request_id,'arguments':arguments},indent=2))
    print(f'fal {endpoint} request: {request_id}',flush=True)
    deadline=time.time()+timeout;seen=0
    for event in handler.iter_events(with_logs=True,interval=2):
        for entry in (getattr(event,'logs',None) or [])[seen:]:
            if status and entry.get('message'):status(f'{tag} · {entry["message"]}')
        seen=len(getattr(event,'logs',None) or [])
        if time.time()>deadline:
            raise TimeoutError(f'fal request {request_id} ({endpoint}) is still running after {timeout}s. Do not resubmit; retrieve it with fal_provider.collect().')
    result=handler.get()
    (folder/f'{tag}-result.json').write_text(json.dumps(result,indent=2))
    return result,request_id

def collect(endpoint,request_id):
    """Fetch the result of an already submitted request (no new charge)."""
    return client().result(endpoint,request_id)
