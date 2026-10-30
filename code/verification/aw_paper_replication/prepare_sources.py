"""Fetch the main-paper AW reconstruction inputs, verifying preserved source hashes."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reproducibility/aw_baseline_reconstruction'
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-dir',type=Path,help='Use already downloaded KEEL ZIPs instead of downloading them.');a=p.parse_args()
 src=OUT/'sources/keel';src.mkdir(parents=True,exist_ok=True)
 spec=json.loads((OUT/'DATASET_SPEC.json').read_text());records=[]
 for item in spec:
  data=(a.dataset_dir/(item['name']+'.zip')).read_bytes() if a.dataset_dir else urllib.request.urlopen(item['url'],timeout=60).read()
  digest=hashlib.sha256(data).hexdigest()
  if digest!=item['archive_sha256']:raise ValueError('Dataset hash mismatch: '+item['name'])
  import io
  with zipfile.ZipFile(io.BytesIO(data)) as z:
   member=item['dat_member'];raw=z.read(member);text=raw.decode('utf-8')
   if hashlib.sha256(text.encode()).hexdigest()!=item['dat_sha256']:raise ValueError('Extracted data hash mismatch')
  (src/(item['name']+'.dat')).write_text(text,encoding='utf-8')
  records.append(dict(dataset=item['name'],sha256=digest))
 subprocess.run([sys.executable,str(ROOT/'code/scripts/fetch_baseline.py')],check=True)
 (OUT/'sources/preparation_checks.json').write_text(json.dumps(records,indent=2));print('Prepared',len(records),'datasets; generated inputs remain ignored by Git.')
if __name__=='__main__':main()
