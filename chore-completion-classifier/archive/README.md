# Archive

Earlier scripts, notebooks and intermediate training sets, kept so that earlier runs and the ideas listed under
"Ideas that did not help" in the project README can be traced (`notebooks/06_other_ideas.ipynb` has the details). The current workflow is the numbered notebooks in the project folder; nothing
here is needed to run them.

| folder / file | what |
|---|---|
| `scripts/` | one script per experiment (augmentation, fine-tuning, patch models, soiled-dish detector, priority rule, bias, crop-top, intake of new photos `add_batch.py`, CDCM helper `overflow_aux.py`, learned bag helper `learned_bag.py`, submission writers) |
| `notebooks/` | earlier comparison notebooks (66 photos, with 11 / 12 / 15 external photos) |
| `data/` | intermediate training-set CSVs; all photos and labels are now in `data/external.csv` |
| `README_old.md` | the earlier project README with the full experiment log and the Kaggle runs table |
| `features_backup_*` | feature caches from before external photos were added |

The scripts were written for the old flat layout (they expect `data/*.csv` and each other in the project folder),
so they need to be copied back there to run. The rule-based open-bag score (`bag_tiles.py`) was removed.
