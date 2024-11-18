# LACA
## Requirements
- python==3.9.18
- NetworkX==2.8.8
- Scipy==1.12.0
- Numpy==1.26.4
- sklearn==0.0.post12
- Third-party library [pyrfm](https://neonnnnn.github.io/pyrfm/)
- Third-party library [LocalGraphClustering](https://github.com/kfoynt/LocalGraphClustering.git)

## Datasets
Get the processed data files from the corresponding `.zip` file and then, unzip and move files to `attr_graphs/<dataset>`.
### Download processed data files
- [cora](https://drive.google.com/file/d/1FyVnpdsTT-lhkVPotUW8OVeuCi1vi3Ey/view?usp=sharing)
- [pubmed](https://drive.google.com/file/d/1DOK3FfslyJoGXUSCSrK5lzdyLfIwOz6k/view?usp=sharing)
- [blogcatalog](https://drive.google.com/file/d/178PqGqh67RUYMMP6-SoRHDoIBh8ku5FS/view?usp=sharing)
- [flickr](https://drive.google.com/file/d/1tZp3EB20fAC27SYWwa-x66_8uGsuU62X/view?usp=sharing)
- [arxiv, reddit, yelp and amazon2m](https://drive.google.com/drive/folders/14h9JrgR1TAVZVXhhl5rZQVcfmyQTR5kL)
### Generate random seed nodes
You may use the randomly generated seed nodes used from our experiments to reproduce our result, or you may also generate them using `attr_graphs/groundtruth.py`.

## Run LACA
```
python src/run.py --data cora --algo laca_c --alpha 0.9 --sigma 1 --dim 64 --e 1e-6
python src/run.py --data cora --algo laca_e --alpha 0.9 --sigma 1 --dim 32 --e 1e-6
```

## Reproductivity
To reproduce our empirical results, run the `run.sh` script.
