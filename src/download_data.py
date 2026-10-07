"""Extract bundled source CSVs (or download missing sources) and verify hashes."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
raw = ROOT / 'data' / 'raw'
raw.mkdir(parents=True, exist_ok=True)

def verify(path, entry):
    if path.stat().st_size != entry['bytes']:
        raise RuntimeError(f"Unexpected source size: {entry['name']}")
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest != entry['sha256']:
        raise RuntimeError(f"Source hash mismatch: {entry['name']}")

for entry in json.loads((ROOT / 'data/source_manifest.json').read_text()):
    target = raw / entry['name']
    if not target.exists():
        temporary = target.with_suffix('.download')
        archive = ROOT / 'data/source' / (entry['name'] + '.gz')
        try:
            if archive.exists():
                with gzip.open(archive, 'rb') as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output)
                verify(temporary, entry)
            else:
                url = 'https://drive.google.com/uc?export=download&id=' + entry['drive_id']
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as output:
                            shutil.copyfileobj(source, output)
                        verify(temporary, entry)
                        break
                    except (OSError, RuntimeError) as error:
                        if attempt == 2:
                            raise RuntimeError(f'Unable to restore {target.name}; download it from the dataset folder.') from error
                        print(f'Retrying {target.name} ({attempt + 1}/3)...', flush=True)
                        time.sleep(2)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    verify(target, entry)
    print('Verified:', target.name)
