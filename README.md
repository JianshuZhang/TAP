# TAP
This repository contains the source code for TAP introduced in the following papers:<br>

* **v1**: [A gru-based encoder-decoder approach with attention for online handwritten mathematical expression recognition](https://arxiv.org/abs/1712.03991)<br>
* **v2**: [Track, attend and parse (TAP): An end-to-end framework for online handwritten mathematical expression recognition](https://ieeexplore.ieee.org/abstract/document/8373726)<br>

Here, **v1** employs the coverage based spatial attention model, **v2** employs the guided hybrid attention model.<br>

## Requirements
* Install [cuda-8.0 cudnn-v7](https://developer.nvidia.com/cudnn)
* Install [Theano.0.10.0](https://github.com/Theano/Theano) with [libgpuarray](https://github.com/Theano/libgpuarray)

## Citation
If you find TAP useful in your research, please consider citing:

	@inproceedings{zhang2017icdar,
      title={A GRU-based Encoder-Decoder Approach with Attention for Online Handwritten Mathematical Expression Recognition},
      author={Jianshu Zhang and Jun Du and Lirong Dai},
      booktitle={International Conference on Document Analysis and Recognition},
      volume={1},
      pages={902--907},
      year={2017}
    }

	
	@article{zhang2019track,
      title={Track, Attend and Parse (TAP): An End-to-end Framework for Online Handwritten Mathematical Expression Recognition},
      author={Zhang, Jianshu and Du, Jun and Dai, Lirong},
      journal={IEEE Transactions on Multimedia},
      volume={21},
      number={1},
      pages={221--233},
      year={2019}
    }

## Description
* Train TAP without using weightnoise and save the best model in terms of WER

      $ bash train.sh
	
* Anneal the best model by using weightnoise and save the new best model

      $ bash train_weightnoise.sh
	
* Reload the new best model and generate the testing latex strings

      $ bash test.sh

## Contact
xysszjs at mail.ustc.edu.cn<br>
West campus of University of Science and Technology of China<br>
Any discussions, suggestions and questions are welcome!
