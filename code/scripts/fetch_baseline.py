"""Fetch the pinned upstream AW-SMOTE implementation into an ignored cache."""
import argparse,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
COMMIT='f0dc14807e02292a096e261ca9af21b0d1300753'
SHA256='72c3d438093538d0c5c3d19bc5dc456be9cdc743a7b907bfec5abd2591d15124'
URL=f'https://raw.githubusercontent.com/guo-source-1/AWSMOTE/{COMMIT}/adw.py'
def main():
 p=argparse.ArgumentParser();p.add_argument('--from-file',type=Path,help='Use an existing downloaded upstream file, verifying the same SHA-256.');a=p.parse_args()
 data=a.from_file.read_bytes() if a.from_file else urllib.request.urlopen(URL,timeout=60).read()
 if hashlib.sha256(data).hexdigest()!=SHA256 and data.startswith(b'# Original AW-SMOTE'):
  data=data.split(b'\n',2)[2]
 if hashlib.sha256(data).hexdigest()!=SHA256:raise SystemExit('Pinned upstream SHA-256 mismatch')
 out=ROOT/'.cache/aw_baseline/aw_smote_original.py';out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data);print('Verified baseline cached at',out)
if __name__=='__main__':main()
