## train
python main.py --bnneck --open-layers 'bottleneck' 'classifier'
## test
python test.py --bnneck --no-pretrained --load-weights 'log/model.pth.tar-60'
