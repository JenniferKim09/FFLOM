# FFLOM Source Code

- Due to the size limit, train data and the pretrained checkpoints can be download from [FFLOM | Zenodo](https://zenodo.org/record/7918738). 

- Users can go to `	./tutorials` fold to find two .ipynb file for the examples of either linker design or R-group design.

## 1. Install Environment

* Please create the environment using `	env.yaml` .
  * The main module used in this code is RDKit and torch, you can also install RDKit by `conda install -c rdkit rdkit` and install torch on the official website. 
  

## 2. Preprocess Dataset

We provide zinc250k dataset / casf dataset / PDBbind dataset in `	./dataset` fold. For example:

- To preprocess zinc_test dataset for linker design task, input codes below in the cmd:  

- ```
  python preprocess.py --data dataset/linker/zinc_test.txt --save_fold ./data_preprocessed/zinc_test_linker/ --name zinc_test --linker_design
  ```

- To preprocess zinc_test dataset for R-group design task, input codes below in the cmd:  

- ```
  python preprocess.py --data dataset/r_group/zinc_test.txt --save_fold ./data_preprocessed/zinc_test_r/ --name zinc_test --r_design
  ```

  - The input dataset should be in the separated SMILES format: 'molecule linker fragments' or 'molecule r-group fragment'. If you only get full molecules (in example.xls), please first use: 
  
  - ```
    python cut_your_own_dataset.py --xls_path example.xls --output_path ./dataset/example.txt --linker_design --n_cores 4 --linker_min 5 --min_path_length 2
    ```
  
  - or R-group case:
  
  - ```
    python cut_your_own_dataset.py --xls_path example.xls --output_path ./dataset/example.txt --r_design --n_cores 4 --fragment_min 5
    ```
  
  - 
  

## 3. Training

We provide pretrained checkpoints in `	./good_ckpt` fold.  To train your own model, use codes like: 

* ```
  python train.py --path ./data_preprocessed/zinc_train/ --batch_size 32 --warm_up --epochs 30 --name train --seed 2019 --all_save_prefix ./
  ```
* To transfer learning on a small dataset (for example, like case studies) using the ZINC250K pretrained checkpoint 306, using codes like:
* ```
  python train.py --path ./data_preprocessed/case/ --batch_size 32 --warm_up --epochs 30 --name xx --seed 2019 --all_save_prefix ./ --init_checkpoint ./good_ckpt/checkpoint306
  ```

  * checkpoints will be saved in `./save_pretrain` fold


## 4. Generation

* Given a checkpoint, you can generate molecules (length of generated fragments equal to the given ones) using codes like: 

* ```
  python generate.py --path ./data_preprocessed/zinc_test_linker/ --gen_out_path ./mols/zinc_test_linker.txt --seed 66666666 --init_checkpoint ./good_ckpt/checkpoint306 --gen_num 10 
  ```

* If you want to design the length of generated fragments by your own, set len_freedom_x and len_freedom_y to become the range of length [len_freedom_x, len_freedom_y]: 

* ```
  python generate.py --path ./data_preprocessed/zinc_test_linker/ --gen_out_path ./mols/test.txt --seed 66666666 --init_checkpoint ./good_ckpt/checkpoint306 --len_freedom_x -1 --len_freedom_y 1 --gen_num 10 
  ```

* 

## 5. Evaluation

If you want to evaluate the generated molecules, using codes for linker design like:

```
python evaluate.py --train_data dataset/linker/zinc_train.txt --gen_data ./mols/test.txt --linker_design --ref_path zinc_250k_valid_test_only.sdf
```

codes for R-group design:

```
python evaluate.py --train_data dataset/r_group/zinc_train.txt --gen_data ./mols/test.txt --r_design --ref_path zinc_250k_valid_test_only.sdf
```







