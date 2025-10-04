# Code for SubNTM

## Preparing libraries
1. Install the following libraries
```bash
numpy==1.26.4
torch==2.7.1
scipy==1.13.1
tqdm==4.67.1
python-dotenv==1.1.1
scikit-learn==1.7.0
gensim==4.3.3
matplotlib==3.10.6
wandb==0.22.1
```
2. Install java
3. Download [this java jar](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to ./evaluations/pametto.jar for evaluating
4. Download and extract [this processed Wikipedia corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to ./datasets/wikipedia/ as an external reference corpus.

## Usage
To run and evaluate our model for 20NG dataset, run this example:
> python main.py --dataset 20NG --num_topic 50

## Acknowledgement
Some part of this implementation is based on [TopMost](https://github.com/BobXWu/TopMost). We also utilizes [Palmetto](https://github.com/dice-group/Palmetto) for the evaluation of topic coherence.