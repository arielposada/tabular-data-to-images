# tabular-data-to-images

```
pip install tf-keras
pip install be-great
pip install tabgan
pip install sentencepiece
```

For deepinsight additionally:
```
python3 -m pip -q install git+https://github.com/alok-ai-lab/pyDeepInsight.git#egg=pyDeepInsight
```


## Basic Setup
```
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

If you are using a 'cuda' capable machine run
```
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```