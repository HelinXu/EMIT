<!--
 * @Author: HelinXu xhl19@mails.tsinghua.edu.cn
 * @Date: 2022-06-23 03:31:05
 * @LastEditTime: 2022-06-23 03:44:09
 * @Description: 
-->
# EMIT: Embedding Matching for Image-Text pairing

Helin Xu (xuhelin1911@gmail.com)

## Requirements

- Pytorch 1.11
- CUDA 11.3

All the packages can be installed by `pip install`.

## How to run

1. Download the data.

2. Unzip the data, and either put the data to `../medium/` or set the `DATA_ROOT` to the unzipped data directory.

3. Split the data into train, val by `python split_data.py`.

4. Train the model: `python train.py --tune_full_model`.

5. Inferer the model: `python inference.py`. The final prediction will be saved to `final_json.json`.