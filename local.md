## 环境安装
1. 安装`experiments/`的依赖环境
    - `pip install -r requirements.txt`有问题, 单独安装的
    - pip安装flash-attn有问题，可以到github下载对应版本, 注意pip安装的pytorch abi=False, flash-attn需要下载abi=False的版本 (abi 即 application binary interface, 如果为True则 会有cxx11的namespace, 在编译时可以通过编译选项来指定)
    - datasets==2.15.0 否则拉取到数据集后解析失败  (服务器上拉取huggingface失败，给配个代理 [代理参考](https://ones.ainewera.com/wiki/#/team/JNwe8qUX/space/HTAfj7mT/page/MwmfGaEG))
    - pip install SentencePiece  #(llama tokenizer需要)
2. 在$QUIK目录下执行
    `pip install -e .`

## 排坑
1. c4数据集下载的坑
参考这个[issue](https://huggingface.co/datasets/allenai/c4/discussions/7), BuilderConfig `allenai--c4` 已经不存在了，删掉这个配置即可
```python
from datasets import load_dataset
traindata = load_dataset(
    'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
)
valdata = load_dataset(
    'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
)
```

## 运行sample
* 下载数据集运行
python llama.py --fp_features_num 256 --model /data/liuhuakai/model/Llama-2-7b-hf --hf_token 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 --dataset c4 --w_bits 4 --w_clip --a_bits 4 --save_qmodel_path local/sim_quant.pt --int8_down_proj --sim_eval --benchmark --load_qmodel_path local/sim_quant.pt

* 模拟运行
python llama.py --fp_features_num 256 --model /data/liuhuakai/model/Llama-2-7b-hf --hf_token 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 --dataset c4 --w_bits 4 --w_clip --a_bits 4 --save_qmodel_path local/sim_quant.pt --int8_down_proj --sim_eval --benchmark --synthetic_data

* linear layer 性能
python layer_benchmark.py



## 记录
* 如何生成act_scales
https://github.com/mit-han-lab/smoothquant/tree/main
atom代码中也有生成act_scales的源代码