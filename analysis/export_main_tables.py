"""Export exact displayed main-paper tables without replacing their experiment provenance."""
from pathlib import Path
import argparse,csv,re
ROOT=Path(__file__).resolve().parents[1]
NAMES=['table1_dataset_summary','table2_groupA_final','table3_groupB_final','table4_starvation_coverage','table5_ablation','table6_transferability','table7_statistical_tests']
HEADERS=[['Subset','Source','Datasets','Samples','Features','Imbalance_ratio','Minority_subgroups','Role'],['Method','wBA','R_G','G_mean','F1','MCC','Precision','Specificity','historical_wBA_rank'],['Method','G_mean','MCC','F1','ROC_AUC','PR_AUC','Recall','Specificity'],['Method','Library','Starved_folds','Starved_percent','Smallest_share_percent','Coverage_percent','Excluded_percent','Majority_area_percent'],['Configuration','wBA','wBA_delta_pp','wBA_lower_datasets','wBA_p_H','R_G','R_G_delta_pp','R_G_p_H','MCC','Starvation_percent','Excluded_percent'],['Method_or_contrast','Starved_percent','Excluded_percent','wBA','R_G','Precision','Specificity','F1','MCC'],['Comparison','Metric','Difference','Win_tie_loss','p','p_H','Conclusion']]
def clean(x):
 x=re.sub(r'\$?\^\{[a-z]\}\$?','',x.strip())
 x=re.sub(r'\\(?:textbf|textit|mathrm|text)\{([^{}]*)\}',r'\1',x)
 for a,b in [(r'\wba','wBA'),(r'\rg','R_G'),(r'\times',' x '),(r'\Delta','Delta'),(r'\%','%'),(r'\_','_'),(r'\,',' '),(r'\quad',''),(r'\scriptsize',''),(r'\ ',' '),('$','')]:x=x.replace(a,b)
 return x.strip()
def export(out):
 text=(ROOT/'paper/main.tex').read_text();tables=re.findall(r'\\begin\{table\}.*?\\end\{table\}',text,re.S);assert len(tables)==7
 out.mkdir(parents=True,exist_ok=True)
 for name,headers,table in zip(NAMES,HEADERS,tables):
  rows=[]
  for line in table.splitlines():
   if '&' in line and line.strip().endswith(r'\\') and r'\multicolumn' not in line and r'\cmidrule' not in line:
    rows.append([clean(x) for x in line.strip()[:-2].split('&')])
  assert len(rows)>1
  rows=rows[1:]
  with (out/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
   writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)
  (out/(name+'.tex')).write_text(table+'\n')
  assert all(len(x)==len(headers) for x in rows),(name,[len(x) for x in rows])
  print(name,len(rows),'rows')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'work/exported_tables');a=p.parse_args();export(a.output)
