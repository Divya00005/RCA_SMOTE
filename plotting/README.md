# Plotting source

Run the four commands in docs/reproduction_instructions.md to generate eight figures under work/generated_figures/. make_fig2.py isolates the Figure 2 block from the retained multi-figure original make_paper_figures.py. Do not run that broad legacy generator to overwrite the final figure set.

make_updated_figures.py reads results/main_tables/table2_groupA_final.csv for the final aggregate means. fig_ablation_analysis.py imports figstyle.py and reads supporting Stage-4 fold evidence. The final canonical PDFs remain the assets used by the manuscript, not automatically replaced by regenerated PDF metadata.
