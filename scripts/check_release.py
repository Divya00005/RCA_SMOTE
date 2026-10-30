"""Check main-paper assets, displayed final means, syntax and release file hashes."""
from pathlib import Path
import ast,csv,hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    checks={}
    tex=(ROOT/'paper/main.tex').read_text()
    checks['no_supplementary_reference_in_main_tex']=not bool(re.search('supplement',tex,re.I))
    refs=re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',tex)
    checks['eight_main_figures_resolve']=len(refs)==8 and all((ROOT/'paper'/p).is_file() for p in refs)
    source=json.loads((ROOT/'reproducibility/SOURCE_MANIFEST.json').read_text())
    figures=[x for x in source['source_assets'] if x['path'].startswith('figures/')]
    checks['canonical_figure_hashes_match_source']=len(figures)==8 and all(sha(ROOT/x['path'])==x['source_sha256'] for x in figures)
    for table,expected in [('table2_groupA_final',{'wBA':.853,'R_G':.674,'F1':.759,'MCC':.719}),('table3_groupB_final',{'G_mean':.693,'F1':.502,'MCC':.457})]:
        rows=list(csv.DictReader((ROOT/'results/main_tables'/(table+'.csv')).open()))
        r=next(x for x in rows if x['Method']=='RCA-SMOTE')
        checks[table+'_final_means']=len(rows)==16 and all(abs(float(r[k])-v)<1e-12 for k,v in expected.items())
    checks['seven_main_csv_tables']=len(list((ROOT/'results/main_tables').glob('*.csv')))==7
    selected=[a for a in source['source_assets'] if a['path'].endswith('.py')]
    checks['all_40_selected_sources_present']=len(selected)==40 and all((ROOT/x['path']).is_file() for x in selected)
    errors=[]
    for folder in ['code','plotting','analysis','scripts','tests']:
        for p in (ROOT/folder).rglob('*.py'):
            try:ast.parse(p.read_text(encoding='utf-8-sig'))
            except Exception as e:errors.append(str(p.relative_to(ROOT))+': '+str(e))
    checks['python_syntax']=not errors
    checks['no_supplementary_documents']=not any(p.name in ['supplementary.tex','supp_tables.tex','RCA_SMOTE_Elsevier_Supplementary.pdf'] for p in ROOT.rglob('*'))
    checks['main_pdf_present']=(ROOT/'paper/RCA_SMOTE_Final.pdf').is_file()
    hashfile=ROOT/'reproducibility/FILE_HASHES.csv'
    if hashfile.exists():
        rows=list(csv.DictReader(hashfile.open()));checks['release_file_hashes']=all((ROOT/r['path']).is_file() and sha(ROOT/r['path'])==r['sha256'] for r in rows)
    print(json.dumps({'checks':checks,'syntax_errors':errors},indent=2))
    if not all(checks.values()):raise SystemExit('Release verification failed')
    print('All release checks passed.')
if __name__=='__main__':main()
