This is the official code for the paper "Explainable Few-shot Knowledge Tracing".

To run the code and baselines, you need to have the following dependencies according to the `requirements.txt`.

Apply your own key of LLMs in LLM_factory\\{LLM_name} and run the `FrcSub-glm4.sh` for a toy example using glm4 on FrcSub dataset.

For the baselines in directory `pyky-baselines`, 
1. Download the dataset and move it to the `pyky-baselines\data\{dataset_name}` directory.
2. Run the dataprocess first to preprocess the dataset.
3. Run the `pyky-baselines\examples\efkt_{dataset_name}_baselines_train.sh` to train the baselines.
4. Run the `pyky-baselines\examples\efkt_{dataset_name}_baselines_predict.sh` to test the baselines.

We collected all the results of the baselines in the `pyky-baselines\examples\efkt_{datasets}_baselines`.

If you find this code useful, please cite the following paper:
```
@article{li2024explainable,
  title={Explainable Few-shot Knowledge Tracing},
  author={Li, Haoxuan and Yu, Jifan and Ouyang, Yuanxin and Liu, Zhuang and Rong, Wenge and Li, Juanzi and Xiong, Zhang},
  journal={arXiv preprint arXiv:2405.14391},
  year={2024}
}
```
