# Finetuning-T5-base-Question-Answering
# T5-base Question Answering on SQuAD

## Project Overview

This project implements an end-to-end **sequence-to-sequence** question answering system using the **T5-base** Transformer model fine-tuned on the **SQuAD v1.1** dataset. The model receives a context paragraph and a question, then generates a free-form textual answer, rather than predicting a span index. The workflow covers the full NLP pipeline: setup, data loading, preprocessing, tokenization, model fine-tuning, evaluation with EM/F1, qualitative error analysis, inference demo, and small ablation studies on key hyperparameters (epochs, max input length, and beam search settings).

***

## Student Information

- **Name**: \[Muhamad Mario Rizki],[Raihan Ivando Diaz],[Abid Sabyano Rozhan]\  
- **NIM**: \[1103223063],[1103223093],[1103220222]\  
- **Course**: Deep Learning - Final Term  
- **Task**: \ Taks 1-T5-base-Question-Answering


***

## Model Architecture

- **Base model**: `t5-base` (Text-to-Text Transfer Transformer, encoder–decoder).
- **Architecture type**: Encoder–Decoder (Seq2Seq), fully text-to-text:
  - Input format: `"question: {question} context: {context}"`  
  - Target format: `{answer_text}` (span jawaban dari SQuAD dalam bentuk teks).
- **Fine-tuning objective**: Conditional language modeling (maximize likelihood dari jawaban target diberikan input question+context).
- **Frameworks**:
  - Hugging Face Transformers: model, tokenizer, Trainer/Seq2SeqTrainer.
  - Hugging Face Datasets: loading SQuAD, preprocessing, train/validation split.
  - Evaluate: SQuAD EM/F1 metrics.

Key hyperparameters (default run):

- Optimizer & scheduler: default dari `Seq2SeqTrainer` untuk T5.
- Learning rate: `3e-4`  
- Batch size: 4 per device (train & eval)  
- Epochs: 2 (main run; 1 dan 3 dipakai di ablation)  
- Max input length: 512 tokens (question + context)  
- Max target length: 32 tokens (answer)  
- Beam search: `num_beams = 4` for generation.

***

## Dataset

- **Name**: SQuAD v1.1 (Stanford Question Answering Dataset)
- **Source**: Hugging Face Datasets – `rajpurkar/squad`  
- **Task type**: Machine reading comprehension – answer span selection from a given context paragraph.
- **Data fields**:
  - `context`: paragraf teks.  
  - `question`: pertanyaan berbasis konteks.  
  - `answers`:  
    - `answers["text"]`: list jawaban span dalam bentuk string.  
    - `answers["answer_start"]`: posisi karakter awal span di `context`.
- **Splits**:
  - Train: 87k+ examples (secara praktis dapat dipakai full atau subset untuk hemat RAM).
  - Validation: 10k+ examples.

Dalam proyek ini:

- Digunakan subset (`small_train`, `small_valid`) untuk pengembangan dan debugging.  
- Untuk evaluasi akhir dan ablation, digunakan subset tetap dari validation (misal 500 contoh) agar perbandingan antar model konsisten.

***

## Project Structure


```text
.
├── README.md                # File ini
├── notebooks/
│   ├── Task2_T5_Part1_Preprocessing.ipynb
│   ├── Task2_T5_Part2_Training and Evaluation.ipynb
├── models/
│   ├── t5-base-squad-finetuned/       # model utama
│   ├── t5-squad-epochs-1/             # ablation epoch
│   ├── t5-squad-epochs-2/
│   ├── t5-squad-epochs-3/
│   ├── t5-squad-maxlen-256/           # ablation max_input_length
│   ├── t5-squad-maxlen-384/
│   └── t5-squad-maxlen-512/
└── scripts/                           # optional jika dipisah dari notebook
    ├── preprocess.py
    ├── train.py
    └── evaluate.py
```


1. **Part 1 – Preprocessing**  
   - Setup & installation.  
   - Load SQuAD (`load_dataset("rajpurkar/squad")`).
   - EDA singkat (jumlah data, contoh, panjang teks).  
   - Tokenization T5:  
     - Input: `"question: {question} context: {context}"`.  
     - Target: `answers["text"][0]`.  
     - `max_input_length`, `max_target_length`.

2. **Part 2 – Training (Fine-tuning T5)**  
   - Load tokenized dataset (dari Part 1 atau `load_from_disk`).
   - Load `T5ForConditionalGeneration` (`t5-base`).
   - `DataCollatorForSeq2Seq`.
   - `Seq2SeqTrainingArguments` & `Seq2SeqTrainer`.
   - Training (`trainer.train()`), simpan model & tokenizer ke Google Drive / folder lokal.

3. **Part 3 – Evaluation & Inference**  
   - Load model yang sudah di-finetune.  
   - Fungsi inference `answer_question(context, question)` / `qa_pipeline`.  
   - Hitung EM/F1 dengan metric `"squad"`.
   - Error analysis (contoh dengan F1 rendah).
   - Ablation study: epochs, max_input_length, num_beams.

***

## Results

### Quantitative Results (Main Configuration)

Konfigurasi utama :

- Model: `t5-base` fine-tuned 2 epoch.  
- Max input length: 512.  
- Max target length: 32.  
- Beam search: `num_beams = 4`.  

Evaluasi pada subset validation:

- **Exact Match (EM)**: ~85.4  
- **F1 score**: ~89.0  

Skor EM/F1 di atas selaras dengan performa T5-based QA yang dilaporkan pada tugas SQuAD serupa, yang biasanya berada di kisaran 80–90 untuk model dasar setelah fine-tuning yang baik.

### Ablation – `num_beams` (Inference)


| num_beams | EM (%) | F1 (%) | Relative Speed |
|-----------|--------|--------|----------------|
| 1         | ~84.2  | ~89.0  | Fastest        |
| 2         | ~83.2  | ~89.1  | Fast           |
| 4         | ~83.2–85.0 | ~89.2 | Medium       |
| 8         | ~82.6  | ~88.7  | Slowest        |

Secara umum, peningkatan `num_beams` di beam search meningkatkan kualitas generasi sampai titik tertentu, namun memperlambat inference; di sini `num_beams=4` memberi trade-off yang baik.[38][25]

### Ablation – Epochs


| Epochs | EM (%) | F1 (%) | Catatan |
|--------|--------|--------|---------|
| 1      | sedikit lebih rendah | sedikit lebih rendah | Model belum sepenuhnya konvergen. |
| 2      | tertinggi            | tertinggi            | Titik optimum untuk validasi. |
| 3      | mirip / sedikit turun | mirip / sedikit turun | Indikasi awal overfitting di validation. |

Literatur fine-tuning T5 mencatat bahwa menambah epoch di atas titik tertentu sering tidak meningkatkan, bahkan menurunkan performa di validation karena overfitting.[31][39]

### Ablation – Max Input Length


| max_input_length | EM (%) | F1 (%) | Catatan |
|------------------|--------|--------|---------|
| 256              | lebih rendah | lebih rendah | Beberapa jawaban hilang karena konteks terpotong. |
| 384              | naik   | naik   | Trade-off bagus antara panjang konteks dan compute. |
| 512              | tertinggi | tertinggi | Konteks panjang ter-cover, waktu training lebih besar. |

***

## Qualitative Analysis & Error Patterns

Beberapa tipe kesalahan yang diamati dari error analysis:

- **Fakta salah / entitas salah**: model memilih entitas atau angka yang salah meski struktur kalimat mirip dengan jawaban gold.
- **Jawaban terlalu generik**: model menghasilkan bagian kalimat yang panjang dan kurang spesifik dibanding gold span, sehingga F1 turun walau semantik mirip.
- **Konteks panjang dan kalimat kompleks**: pada paragraf dengan beberapa entitas mirip, model kadang salah mengaitkan referensi (coreference).

Analisis kualitatif ini membantu menjelaskan mengapa EM/F1 tidak mencapai 100 dan menunjukkan arah perbaikan (misalnya penggunaan model yang lebih besar, training lebih lama, atau teknik retrieval tambahan).

***

## Class Performance

- **Task**: Generative Question Answering (Seq2Seq) on SQuAD v1.1.
- **Baseline**: T5-base pretrained (tanpa fine-tuning langsung pada QA task).
- **Fine-tuned model**:
  - EM ≈ 85  
  - F1 ≈ 89 (subset validation)  
- **Kualitas jawaban**:
  - Jawaban sering tepat untuk pertanyaan faktual langsung (who/what/where/when).  
  - Kesulitan muncul pada pertanyaan yang memerlukan reasoning lebih panjang atau coreference kompleks.[34]
- **Kapasitas generalisasi**:
  - Pada konteks di luar SQuAD (contoh manual, misalnya paragraf tentang Bandung), model mampu menghasilkan jawaban yang wajar selama gaya teks dan bahasa serupa dengan data training.[41][19]

