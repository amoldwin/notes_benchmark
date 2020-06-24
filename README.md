# notes_benchmark
MIMIC-III Benchmarks experiments incorporating clinical notes  
  
Creating the benchmark will follow the same instructions as the original MIMIC-III Benchmark, with one or two devitions.  
  
Run the scripts in the following order:  
  
`python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/` (We also make use of two derived data files not present in the MIMIC-III database itself. These will be available at TBA) in `mimic3csv.py` you will need to edit the paths `words_path` and `cuis_path` to point to these files.
  
`python -m mimic3benchmark.scripts.validate_events data/root/`  
  
`python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/`  
  
`python -m mimic3benchmark.scripts.split_train_and_test data/root/`  
  
The next five commands may be run concurrently:  
`python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/  
python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/  
python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/  
python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/  
python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/`  

`python -m mimic3models.split_train_val data/in-hospital-mortality `  
`python -m mimic3models.split_train_val data/decompensation `  
`python -m mimic3models.split_train_val data/phenotyping `  
`python -m mimic3models.split_train_val data/length-of-stay `  
`python -m mimic3models.split_train_val data/multitask`  

Finally, we add one script that is not part of the original benchmark, to ensure that all episodes contain at least one text event and one structured event:  
`python -m mimic3benchmark.create_filtered_listfiles`  

We train and test models using the same instructions as the original MIMIC-III Benchmark, but we require the inclusions of new arguments to specify what data to use. For example, the following would train an LSTM model on only structured data.  
  
 `python -um mimic3models.in_hospital_mortality.main --network mimic3models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic3models/in_hospital_mortality --structured`  
   
 The inclusion of the `--weighted` argument will add class weights to the training to counteract a class imbalance (available only for decompensation and in-hospital mortality)  
   
 The inclusion of `--condensed` will remove all timesteps in where no notes are present and infer structured data values at those timesteps based on their most recent recorded value.  
   
 The inclusion of `--doc2vec`, `--words`, and `--cuis` will cause the model to train on a doc2vec document encoding, a bag of words, or a bag of UMLS concepts respectively.  
   
 Note that `--structured --doc2vec` can be used to train teh LSTM on both structured data and doc2vec note encodings simultaneously, but no other combinations are allowed. This is because the bag of words/concepts required an embedding layer absent in the original baseline LSTM model.  
   
 
