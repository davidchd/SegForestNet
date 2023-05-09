# SegForestNet
Submodule of [BU School of Public Health - Greenspace Ecosystem Services in K-12 schools](https://github.com/BU-Spark/ml-busph-greenspace/)

## Original Repository
Please refer to [gritzner/SegForestNet](https://github.com/gritzner/SegForestNet) for details of SegForestNet.

## Environment
1. Create a new conda virtual environment:
    ```sh
    conda env create -f environment.yml
    ```
2. Follow instructions in the original repository to create a configuration file called ```~/.aethon/user.yaml``` 
3. Compile the code written in Rust:
    ```sh
    python aethon.py semseg potsdam 0 SegForestNet MobileNetv2 --compile
    ```

## Result
The model was trained on Toulouse dataset. Files of ifferent version of the pretrained weights are saved in [```./models/pretrain/```](./models/pretrain/), where filenames are in ```Epoch_BatchSize_ModelInputShape.pt```. Some results are shown in the 3 jupyter notebooks of the current directory. Among them, the best result we have is [```greenspace-deploy-50m-origin.ipynb```](./greenspace-deploy-50m-origin.ipynb), while [```greenspace-deploy-50m.ipynb```](./greenspace-deploy-50m.ipynb) and [```greenspace-deploy-100m.ipynb```](./greenspace-deploy-100m.ipynb) are some more experiments.

## Train
1. Download datasets in ```./tmp/```
2. Modify any ```DatasetLoader``` class in ```./datasets/``` as needed 
3. Make any necessary changes to ```./cfgs/semseg.yaml```
4. Start training the model by 
    ```sh 
    python aethon.py semseg Toulouse 0 SegForestNet MobileNetv2
    ```

## Evaluate
1. Unzip the compressed file of images to ```./tmp/Greenspace``` 
2. In ```./tmp/Greenspace``` directory, replacing the spaces in the filename is necessary:
    ```sh 
    find . -name '*.png' -exec sh -c 'mv "$0" "${0// /_}"' {} \; 
    ```
3. Make any necessary changes to [```eval.py```](./eval.py) or any of the ```eval_*.py``` files 
4. Evaluation the model on Greenspace dataset by:
   ```sh
   python eval.py semseg greenspace 0 SegForestNet MobileNetv2 --cpu
   ```
   Results would be generated into [```./tmp/Greenspace/result/```](./tmp/Greenspace/result/) directory. Refer to [```greenspace-deploy-50m-origin.ipynb```](./greenspace-deploy-50m-origin.ipynb) for more details of the generated results. 
