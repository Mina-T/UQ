import yaml

with open("Latte_config.yaml") as f:
    cfg = yaml.safe_load(f)


n_ensembles=
for i in {1..$n_ensembles}; do mkdir ensembel_$i; cp Latte_train.ini  sample.job  ensembel_$i; cd ensembel_$i; sbatch  sample.job  ; cd ../; done
