# Finetuning-T5-base-Question-Answering
# T5-base Question Answering on SQuAD

## Project Overview

This project implements an end-to-end **sequence-to-sequence** question answering system using the **T5-base** Transformer model fine-tuned on the **SQuAD v1.1** dataset. The model receives a context paragraph and a question, then generates a free-form textual answer, rather than predicting a span index. The workflow covers the full NLP pipeline: setup, data loading, preprocessing, tokenization, model fine-tuning, evaluation with EM/F1, qualitative error analysis, inference demo, and small ablation studies on key hyperparameters (epochs, max input length, and beam search settings).[14][15][16][17]

***

## Student Information

- **Name**: \<Isi nama kamu\>  
- **NIM**: \<Isi NIM\>  
- **Course**: Deep Learning / NLP  
- **Institution**: \<Nama kampus\>  
- **Location**: Bandung, West Java, Indonesia  

Silakan sesuaikan bagian ini sebelum upload ke GitHub.

***

## Model Architecture

- **Base model**: `t5-base` (Text-to-Text Transfer Transformer, encoder–decoder).[15][18]
- **Architecture type**: Encoder–Decoder (Seq2Seq), fully text-to-text:
  - Input format: `"question: {question} context: {context}"`  
  - Target format: `{answer_text}` (span jawaban dari SQuAD dalam bentuk teks).[19][20]
- **Fine-tuning objective**: Conditional language modeling (maximize likelihood dari jawaban target diberikan input question+context).[18]
- **Frameworks**:
  - Hugging Face Transformers: model, tokenizer, Trainer/Seq2SeqTrainer.[21][18]
  - Hugging Face Datasets: loading SQuAD, preprocessing, train/validation split.[22][14]
  - Evaluate: SQuAD EM/F1 metrics.[17][23]

Key hyperparameters (default run):

- Optimizer & scheduler: default dari `Seq2SeqTrainer` untuk T5.[21]
- Learning rate: `3e-4`  
- Batch size: 4 per device (train & eval)  
- Epochs: 2 (main run; 1 dan 3 dipakai di ablation)  
- Max input length: 512 tokens (question + context)  
- Max target length: 32 tokens (answer)  
- Beam search: `num_beams = 4` for generation.[24][25]

***

## Dataset

- **Name**: SQuAD v1.1 (Stanford Question Answering Dataset)[26][14]
- **Source**: Hugging Face Datasets – `rajpurkar/squad`  
- **Task type**: Machine reading comprehension – answer span selection from a given context paragraph.[27][26]
- **Data fields**:
  - `context`: paragraf teks.  
  - `question`: pertanyaan berbasis konteks.  
  - `answers`:  
    - `answers["text"]`: list jawaban span dalam bentuk string.  
    - `answers["answer_start"]`: posisi karakter awal span di `context`.[28][27]
- **Splits**:
  - Train: 87k+ examples (secara praktis dapat dipakai full atau subset untuk hemat RAM).[14]
  - Validation: 10k+ examples.[14]

Dalam proyek ini:

- Digunakan subset (`small_train`, `small_valid`) untuk pengembangan dan debugging.  
- Untuk evaluasi akhir dan ablation, digunakan subset tetap dari validation (misal 500 contoh) agar perbandingan antar model konsisten.[17][22]

***

## Project Structure

Sesuaikan nama file dengan repository-mu; struktur umum yang disarankan:

```text
.
├── README.md                # File ini
├── notebooks/
│   ├── Task2_T5_Part1_Preprocessing.ipynb
│   ├── Task2_T5_Part2_Training.ipynb
│   └── Task2_T5_Part3_Eval_Inference.ipynb
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

Ringkasan isi notebook utama:

1. **Part 1 – Preprocessing**  
   - Setup & installation.  
   - Load SQuAD (`load_dataset("rajpurkar/squad")`).[14]
   - EDA singkat (jumlah data, contoh, panjang teks).  
   - Tokenization T5:  
     - Input: `"question: {question} context: {context}"`.  
     - Target: `answers["text"][0]`.  
     - `max_input_length`, `max_target_length`.[20][19]

2. **Part 2 – Training (Fine-tuning T5)**  
   - Load tokenized dataset (dari Part 1 atau `load_from_disk`).[22]
   - Load `T5ForConditionalGeneration` (`t5-base`).[18]
   - `DataCollatorForSeq2Seq`.[29]
   - `Seq2SeqTrainingArguments` & `Seq2SeqTrainer`.[30][31]
   - Training (`trainer.train()`), simpan model & tokenizer ke Google Drive / folder lokal.[32][33]

3. **Part 3 – Evaluation & Inference**  
   - Load model yang sudah di-finetune.  
   - Fungsi inference `answer_question(context, question)` / `qa_pipeline`.  
   - Hitung EM/F1 dengan metric `"squad"`.[23][17]
   - Error analysis (contoh dengan F1 rendah).[34]
   - Ablation study: epochs, max_input_length, num_beams.[25][35]

***

## Results

### Quantitative Results (Main Configuration)

Konfigurasi utama (contoh, sesuaikan dengan hasilmu):

- Model: `t5-base` fine-tuned 2 epoch.  
- Max input length: 512.  
- Max target length: 32.  
- Beam search: `num_beams = 4`.  

Evaluasi pada subset validation (misal 500 contoh):

- **Exact Match (EM)**: ~85.4  
- **F1 score**: ~89.0  

Skor EM/F1 di atas selaras dengan performa T5-based QA yang dilaporkan pada tugas SQuAD serupa, yang biasanya berada di kisaran 80–90 untuk model dasar setelah fine-tuning yang baik.[16][36][37]

### Ablation – `num_beams` (Inference)

Contoh hasil (isi sesuai hasil loop-mu):

| num_beams | EM (%) | F1 (%) | Relative Speed |
|-----------|--------|--------|----------------|
| 1         | ~84.2  | ~89.0  | Fastest        |
| 2         | ~83.2  | ~89.1  | Fast           |
| 4         | ~83.2–85.0 | ~89.2 | Medium       |
| 8         | ~82.6  | ~88.7  | Slowest        |

Secara umum, peningkatan `num_beams` di beam search meningkatkan kualitas generasi sampai titik tertentu, namun memperlambat inference; di sini `num_beams=4` memberi trade-off yang baik.[38][25]

### Ablation – Epochs

Contoh pola yang biasanya muncul:

| Epochs | EM (%) | F1 (%) | Catatan |
|--------|--------|--------|---------|
| 1      | sedikit lebih rendah | sedikit lebih rendah | Model belum sepenuhnya konvergen. |
| 2      | tertinggi            | tertinggi            | Titik optimum untuk validasi. |
| 3      | mirip / sedikit turun | mirip / sedikit turun | Indikasi awal overfitting di validation. |

Literatur fine-tuning T5 mencatat bahwa menambah epoch di atas titik tertentu sering tidak meningkatkan, bahkan menurunkan performa di validation karena overfitting.[31][39]

### Ablation – Max Input Length

Contoh pola:

| max_input_length | EM (%) | F1 (%) | Catatan |
|------------------|--------|--------|---------|
| 256              | lebih rendah | lebih rendah | Beberapa jawaban hilang karena konteks terpotong. |
| 384              | naik   | naik   | Trade-off bagus antara panjang konteks dan compute. |
| 512              | tertinggi | tertinggi | Konteks panjang ter-cover, waktu training lebih besar. |

Studi pada QA berbasis konteks panjang menunjukkan bahwa pemotongan konteks agresif dapat menurunkan akurasi, terutama untuk pertanyaan yang jawabannya berada jauh di dalam paragraf.[35][26]

***

## Qualitative Analysis & Error Patterns

Beberapa tipe kesalahan yang diamati dari error analysis:

- **Fakta salah / entitas salah**: model memilih entitas atau angka yang salah meski struktur kalimat mirip dengan jawaban gold.[34]
- **Jawaban terlalu generik**: model menghasilkan bagian kalimat yang panjang dan kurang spesifik dibanding gold span, sehingga F1 turun walau semantik mirip.[16]
- **Konteks panjang dan kalimat kompleks**: pada paragraf dengan beberapa entitas mirip, model kadang salah mengaitkan referensi (coreference).[40][34]

Analisis kualitatif ini membantu menjelaskan mengapa EM/F1 tidak mencapai 100 dan menunjukkan arah perbaikan (misalnya penggunaan model yang lebih besar, training lebih lama, atau teknik retrieval tambahan).

***

## Class Performance

Bagian ini bisa kamu gunakan di laporan / README untuk merangkum performa akhir model:

- **Task**: Generative Question Answering (Seq2Seq) on SQuAD v1.1.[26][14]
- **Baseline**: T5-base pretrained (tanpa fine-tuning langsung pada QA task).[18]
- **Fine-tuned model**:
  - EM ≈ 85  
  - F1 ≈ 89 (subset validation)  
- **Kualitas jawaban**:
  - Jawaban sering tepat untuk pertanyaan faktual langsung (who/what/where/when).  
  - Kesulitan muncul pada pertanyaan yang memerlukan reasoning lebih panjang atau coreference kompleks.[34]
- **Kapasitas generalisasi**:
  - Pada konteks di luar SQuAD (contoh manual, misalnya paragraf tentang Bandung), model mampu menghasilkan jawaban yang wajar selama gaya teks dan bahasa serupa dengan data training.[41][19]

