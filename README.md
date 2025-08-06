# CGP-Tuning: Structure-Aware Soft Prompt Tuning for Code Vulnerability Detection

This project implements **CGP-Tuning**, a graph-enhanced soft prompt tuning method designed for **code vulnerability detection**. The approach incorporates graph-based structural information of source code into prompt tuning, enhancing the detection capabilities for security vulnerabilities in code.

The dataset used in this project is **DiverseVul**, which can be accessed here: https://github.com/wagner-group/diversevul.

## Project Structure

The project directory is organized as follows:

- `templates/`: Contains the templates used for inference.
- `tuners/`: Contains the key implementation of CGP-Tuning.
- `utils/`: Includes utility scripts for data preprocessing.
- `example.json`: An example data sample for running.
- `main.py`: The main script that demonstrates how to run CGP-Tuning.

## Requirement

To run this project, ensure that the following dependencies are installed with the specified versions to ensure compatibility and optimal performance:

- **python**: v3.11.9
- **transformers**: [v4.46.1](https://github.com/huggingface/transformers)
- **trl**: [v0.11.0](https://github.com/huggingface/trl)
- **peft**: [v0.13.2](https://github.com/huggingface/peft)
- **pytorch**: [v2.4.0](https://pytorch.org/)
- **cuda toolkit**: v11.8.0  
- **cuDNN**: v8.9.2.26

## Citation

If you find this work useful, please cite our paper:

```bibtex
@ARTICLE{feng2025cgptuning,
  author={Feng, Ruijun and Pearce, Hammond and Liguori, Pietro and Sui, Yulei},
  journal={IEEE Transactions on Software Engineering}, 
  title={CGP-Tuning: Structure-Aware Soft Prompt Tuning for Code Vulnerability Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TSE.2025.3591934}
}
```
