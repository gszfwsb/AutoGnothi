<div align="center">
<h2>
Gnothi Seauton | Γνώθι Σαυτόν:

Empowering Faithful Self-Interpretability in Black-Box Transformers
</h2>
</div>


This repository contains a reference implementation for **ICLR 2025** paper **_Gnothi Seauton: Empowering Faithful Self-Interpretability in Black-Box Transformers_**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/pipeline_light.png">
  <source media="(prefers-color-scheme: light)" srcset="./assets/pipeline_light.png">
  <img alt="FunCoder method diagram." src="./assets/method_light.png">
</picture>

<br />

> The debate between self-interpretable models and post-hoc explanations for black
> box models is central to Explainable AI (XAI). Self-interpretable models, such
> as concept-based networks, offer insights by connecting decisions to human
> understandable concepts but often struggle with performance and scalability.
> Conversely, post-hoc methods like Shapley values, while theoretically robust, are
> computationally expensive and resource-intensive. To bridge the gap between
> these two lines of research, we propose a novel method that combines their
> strengths, providing theoretically guaranteed self-interpretability for black-box
> models without compromising prediction accuracy. Specifically, we introduce a
> parameter-efficient pipeline, AutoGnothi, which integrates a small side network
> into the black-box model, allowing it to generate Shapley value explanations with
> out changing the original network parameters. This side-tuning approach
> significantly reduces memory, training, and inference costs, outperforming traditional
> parameter-efficient methods, where full fine-tuning serves as the optimal baseline.
> AutoGnothi enables the black-box model to predict and explain its predictions
> with minimal overhead. Extensive experiments show that AutoGnothi offers accurate
> explanations for both vision and language tasks, delivering superior computational
> efficiency with comparable interpretability.

## 🔧 Setup

### Environment

```bash
conda create -y -n autognothi python=3.8
conda activate autognothi
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

### Resources

No API key or any of the sort are required for this project. However, you do need to setup the pre-trained parameters and datasets first. Run the following command under repo root and the script will automatically download & pre-process all required datasets and pre-trained models, which should require no more than 10 GiB disk space.

```bash
python ./main.py preload_all
```

## 🧪 Experiments

To run an AutoGnothi experiment, you should create a directory named after your experiment, with a properly-formed `.hparams.json` inside. All logs, checkpoints and reports will be saved to this directory. You can find some example experiments in the `/experiments` directory that can get you quickly started with the process.

To start training and evaluating, you need to run the following commands:

```bash
# fine-tune base models into models adaptive to certain tasks
python ./main.py pretrain_classifier ./experiments/ft_bert_base_tayp --device cuda:0
# train & measure each method in the experiment
python ./main.py train_all ./experiments/bert_base_tayp_vanilla --device cuda:0
python ./main.py measure_all ./experiments/bert_base_tayp_vanilla --device cuda:0
```

If you're looking for more fine-grained task control, you can find these commands individually in `main.py`'s help message. We have single commands for each task stage, e.g. `train_classifier`, `train_surrogate`, `train_explainer`, and so on. The evaluation reports are located under `/$path/$to/$experiment/.reports/`, and you can use tools in `playground/` to read them in batch.

## 📝 Citation

If you find our work helpful, you can cite this paper as:

```bibtex
@inproceedings{
    wang2025gnothi,
    title={Gnothi Seauton: Empowering Faithful Self-Interpretability in Black-Box Models},
    author={Shaobo Wang and Hongxuan Tang and Mingyang Wang and Hongrui Zhang and Xuyang Liu and Weiya Li and Xuming Hu and Linfeng Zhang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=UvMSKonce8}
}
```

## 📫 Contact
- [Shaobo Wang](shaobowang1009@sjtu.edu.cn) (Shanghai Jiao Tong University)  
- [Hongxuan Tang](jeffswt@outlook.com)

## 💾 Contributing

We're open to pull requests that adds more new features, pre-trained models and datasets. Bug fixes and other improvements are also welcomed. If you have any questions, feel free to contact the authors directly.
