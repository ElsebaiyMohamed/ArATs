# Arabic Text Summarization using Transformers: LANS & WikiLingua Datasets

## Introduction

Summarization plays a vital role in natural language processing, helping distill essential information from large text volumes. To tackle this challenge, we've developed a transformer-based summarization model. Transformers, designed for tasks with long-range dependencies, excel in abstractive summarization—generating summaries from scratch.

Our work leverages the LANS (Large-scale Arabic News Summarization Corpus) dataset and the WikiLingua dataset, comprising ~770k article-summary pairs across 18 languages from WikiHow. With a focus on abstractive techniques, we aim to generate high-quality summaries.

## LANS Dataset

The LANS dataset offers a rich collection of Arabic news articles and summaries. It encompasses 8,443,484 articles from 22 renowned Arab newspapers spanning two decades (1999-2019). Each article is accompanied by a journalist-written summary, capturing the core content.

**Dataset Statistics:**

- Total Articles: 8,443,484
- Articles Range: 22 newspapers
- Largest Subset: 1,004,893 articles
- Smallest Subset: 128,785 articles
- Categories: Politics, Economy, Sports, Health, Technology
- Languages: Mainly Modern Standard Arabic (MSA), plus dialects like Moroccan, Egyptian, and Saudi Arabian

**Evaluation:**
Automatic evaluation using ROUGE, METEOR, and BLEU metrics yielded promising results:

- ROUGE-1: 0.5310
- METEOR: 0.5460
- BLEU: 0.4740

Human evaluation of 1000 summaries confirmed high quality, averaging a score of 4.5.

## WikiLingua Dataset

For training and evaluation, we employed the WikiLingua dataset with ~770k article-summary pairs from WikiHow. The dataset includes parallel article-summary pairs in English, enabling cross-language training and assessment.

## Achievements

Our work has yielded significant achievements, including:

- **18th Rank** among 100 competitors in competitive Abstractive Arabic Text Summarization.
- Achieving a **ROUGE score of 19.2** on Arabic text summarization.
- Leveraging the **LANS dataset** and optimizing model training through efficient word pre-segmentation.
- Collaborating on refining data preprocessing pipelines for offline and streaming contexts.
- Utilizing pretrained models (MT5, Bart, araBart, araBert, Bert2GPT, Bert2Bert) via TensorFlow and Huggingface, coupled with original generative strategies for enhanced summarization.
- Gaining experience in text preprocessing, research, co-working, hyperparameters tuning, and problem-solving.

## Future Work

Our roadmap for future enhancements includes:

- Further enhancing model performance through hyperparameter tuning and novel techniques.
- Exploring the incorporation of additional features and multi-modal inputs.
- Extending our model's utility to summarization tasks in various languages beyond Arabic.

## Conclusion

In this consolidated readme, we've introduced our journey in Arabic text summarization using transformer models. We've discussed the valuable LANS dataset, our summarization approach, evaluation metrics, achievements, and plans for the future. Our commitment to advancing summarization technology and enabling machines to effectively distill text data remains unwavering.
