# vq-vae-test
### connecting to Colab on vscode
```python
# on Colab
!pip install colab_ssh --upgrade

from colab_ssh import launch_ssh_cloudflared, init_git_cloudflared
launch_ssh_cloudflared(password="<PASSWORD>")

# follow "Client machine configuration" at first time
# follow "VSCode Remote SSH" guide
```

### 실행 방법
```python
$ python train.py -cn local # local setting
$ python train.py -cn colab # colab setting
$ python train.py -cn paper # paper setting
```

#### referenced from Aäron van den Oord's work
* https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
