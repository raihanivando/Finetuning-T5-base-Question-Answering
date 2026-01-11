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

## Model Architecture

Model utama yang digunakan adalah **T5-base**, sebuah encoder-decoder Transformer untuk task generik text-to-text. 

- Checkpoint: `t5-base` dari Hugging Face Transformers. 
- Komponen utama:
  - Encoder: stack beberapa `T5Block` dengan self-attention dan feed-forward layer. 
  - Decoder: stack `T5Block` dengan self-attention, cross-attention ke encoder, dan feed-forward layer. 
  - Shared embedding antara encoder dan decoder (`shared.Embedding`). 
  - Output head: linear layer (`lm_head`) yang memetakan hidden states ke vocabulary size (~32.128 token). 
- Ukuran model:
  - Hidden size: 768.
  - Feed-forward dimension: 3072.
  - Jumlah layer encoder dan decoder: 12 blok. 

Arsitektur ini difinetune secara end-to-end pada SQuAD dengan objective sequence-to-sequence, di mana input berupa teks “question ... context ...” dan target berupa teks jawaban. 

---

## Dataset

### Deskripsi SQuAD

Proyek menggunakan **SQuAD v1.1** yang dimuat melalui `datasets.load_dataset("rajpurkar/squad")`. 

- Split dataset:
  - Train: 87.599 contoh. 
  - Validation: 10.570 contoh. 
- Fitur per contoh:
  - `id`: identitas unik untuk setiap Q&A pair. 
  - `title`: judul artikel Wikipedia. 
  - `context`: paragraf teks di mana jawaban berada. 
  - `question`: pertanyaan natural language. 
  - `answers`: struktur dengan `text` dan `answer_start`. 

### Subset untuk Eksperimen

Untuk mengurangi beban komputasi di Colab, proyek menggunakan subset data: 

- `small_train`: 5.000 contoh pertama dari train.
- `small_valid`: 1.000 contoh pertama dari validation.

Subset ini digunakan untuk:
- Eksperimen awal preprocessing dan tokenisasi.
- Training utama dan beberapa ablation dengan resource GPU terbatas. 

### Preprocessing dan Tokenisasi

Dataset diubah ke format text-to-text sebagai berikut: 

- Input:
  - Format string: `"question {question} context {context}"`.
- Target:
  - Jawaban pertama `answers["text"][0]` jika tersedia, atau string kosong jika tidak ada jawaban. 

Tokenisasi menggunakan `T5TokenizerFast` dengan parameter: 

- `max_input_length`: dicoba beberapa nilai seperti 256, 384, dan 512 untuk keperluan ablation. 
- `max_target_length`: sekitar 32 token. 
- Padding: `padding="max_length"`.
- Truncation: `True` untuk memotong input yang terlalu panjang. 

Hasil tokenisasi (`input_ids`, `attention_mask`, dan `labels`) kemudian disimpan ke Google Drive agar dapat di-load kembali pada tahap training. 

---

## Project Structure

Struktur logis proyek (berdasarkan dua notebook utama) adalah sebagai berikut: 

```text
Finetuning-T5-base-Question-Answering/
├── README.md # Project documentation
├── notebooks/ # Jupyter notebooks
│ ├── Task2_T5_QA_SQuAD_Part1_Preprocessing.ipynb # Data loading, preprocessing, tokenization
│ ├── Task2_T5_QA_SQuAD_Part1_Training_and_Evaluation.ipynb # Training and Evaluation
└── Reports

```

### 1. `Task2_T5_QA_SQuAD_Part1_Preprocessing.ipynb`

Notebook ini menangani tahap awal: 

- Setup lingkungan:
  - Install library (`transformers`, `datasets`, `accelerate`, `evaluate`, `sentencepiece`). 
  - Cek PyTorch dan CUDA. 
- Load dataset:
  - `load_dataset("rajpurkar/squad")` untuk train dan validation. 
  - Buat subset `small_train` (5k) dan `small_valid` (1k). 
- Exploratory Data Analysis (EDA):
  - Analisis panjang pertanyaan dan konteks (contoh distribusi panjang). 
  - Menampilkan beberapa contoh Q&A. 
- Preprocessing:
  - Definisi fungsi `build_preprocess_fn(tokenizer, max_input_length, max_target_length)` untuk membuat fungsi preprocess berbasis parameter panjang input dan target. 
  - Menggabungkan question dan context ke dalam satu string input, serta menyiapkan target jawaban. 
- Tokenisasi dan penyimpanan:
  - `tokenized_train = small_train.map(preprocess_fn, ...)`.
  - `tokenized_valid = small_valid.map(preprocess_fn, ...)`. 
  - Menyimpan hasil ke disk (Google Drive) menggunakan `save_to_disk`. 

### 2. `Task2_T5_QA_SQuAD_Part1_Training_and_Evaluation.ipynb`

Notebook ini fokus pada pemanfaatan dataset ter-tokenisasi untuk training, evaluasi, dan ablation study. 

- Setup & load:
  - Mount Google Drive dan load dataset tokenized dari folder yang sudah dibuat di Part 1. 
- Model & Data Collator:
  - Load `T5ForConditionalGeneration` dan `T5TokenizerFast` dari checkpoint `t5-base`. 
  - Definisikan `DataCollatorForSeq2Seq` untuk dynamic padding. 
- Training core:
  - Definisikan `Seq2SeqTrainingArguments` (batch size, learning rate, epoch, logging, fp16). 
  - Inisialisasi `Seq2SeqTrainer` dan jalankan `trainer.train()`. 
  - Simpan model dan tokenizer ke local path dan ke Google Drive. 
- Evaluasi:
  - Gunakan metrik SQuAD (`evaluate.load("squad")`) untuk EM dan F1 pada subset validation. 
  - Definisikan helper `evaluate_model(model_dir, val_subset)` untuk mengevaluasi model yang disimpan. 
- Ablation & Analisis:
  - Eksperimen variasi `num_beams`, `max_input_length`, dan `num_train_epochs`. 
  - Error analysis untuk contoh dengan skor F1 rendah. 
- Demo inference:
  - Fungsi `answer_question(context, question)` dan `qa_pipeline(context, question, num_beams)` untuk uji coba interaktif. 

---

## Result

Bagian ini merangkum hasil training dan evaluasi utama yang diperoleh dari eksperimen pada subset SQuAD. 

### Metrik Utama (EM & F1)

Evaluasi dilakukan menggunakan metrik resmi SQuAD pada subset validation (misalnya 500 contoh) untuk model T5 yang telah difinetune. 

- Konfigurasi baseline (contoh umum):
  - Model: `t5-base` difinetune pada 5.000 contoh train.
  - `num_train_epochs`: 2.
  - `num_beams`: 4 untuk inference baseline. 
- Hasil metrik pada subset:
  - Exact Match (EM): sekitar **83–84%**. 
  - F1 score: sekitar **89–90%**. 

Notebook juga menampilkan hasil evaluasi dengan beberapa nilai `num_beams`, misalnya: 

- `num_beams = 1`:
  - EM ≈ 84.20
  - F1 ≈ 89.86
  - Waktu inference ~11–13 detik pada subset yang sama. 
- `num_beams = 2`:
  - EM ≈ 83.20
  - F1 ≈ 89.18
  - Waktu inference lebih lambat dibanding `num_beams=1`. 
- `num_beams = 4`:
  - EM ≈ 83.20
  - F1 ≈ 89.29
  - Waktu inference ~27–30 detik. 
- `num_beams = 8`:
  - EM sedikit turun ke kisaran 82–83%.
  - F1 turun ke kisaran 88–89%.
  - Waktu inference meningkat signifikan (~40–50 detik). 

Hasil ini menunjukkan trade-off antara kualitas dan kecepatan inference, dengan `num_beams=1` sudah cukup kompetitif sekaligus paling efisien. 
## Class Performance

Bagian ini merangkum performa model dalam konteks “kelas” konfigurasi atau setting eksperimen yang berbeda, bukan kelas label seperti klasifikasi. 

### Performa berdasarkan konfigurasi decoding (`num_beams`)

| Konfigurasi (kelas)        | EM (±) | F1 (±) | Waktu Inference (subset) | Catatan                                                                 |
|----------------------------|--------|--------|---------------------------|-------------------------------------------------------------------------|
| `num_beams = 1`           | ~84.2  | ~89.9  | ~11–13 s                  | Paling cepat, performa sudah sangat baik.                       |
| `num_beams = 2`           | ~83.2  | ~89.2  | Lebih lambat dari 1      | Tidak memberi peningkatan signifikan.                           |
| `num_beams = 4`           | ~83.2  | ~89.3  | ~27–30 s                  | Digunakan sebagai konfigurasi baseline.                         |
| `num_beams = 8`           | ~82–83 | ~88–89 | ~40–50 s                  | Kualitas sedikit menurun, waktu meningkat drastis.             |

### Performa berdasarkan panjang input (`max_input_length`)

Notebook menyiapkan eksperimen untuk beberapa kelas panjang input: 

- `max_input_length = 256`
- `max_input_length = 384`
- `max_input_length = 512`

Untuk setiap kelas panjang input:
- Dataset di-tokenisasi ulang sesuai `max_input_length`. 
- Model dilatih 1 epoch dan disimpan di direktori `t5-squad-maxlen-{maxlen}`. 
- Struktur `ablation_maxlen` dibuat untuk menyimpan EM dan F1 per konfigurasi, meskipun sebagian nilai masih placeholder pada snapshot. 

Secara intuitif, panjang input lebih besar memungkinkan konteks yang lebih lengkap, tetapi dengan biaya waktu training dan inference yang lebih tinggi. 

### Performa berdasarkan jumlah epoch (`num_train_epochs`)

Terdapat fungsi helper untuk melatih model dengan kelas epoch berbeda: 1, 2, dan 3 epoch. 

- Direktori output:
  - `t5-squad-epochs-1`
  - `t5-squad-epochs-2`
  - `t5-squad-epochs-3` 
- Struktur `ablation_epochs` menggambarkan rencana pencatatan EM dan F1 untuk masing-masing kelas epoch, walaupun tidak semua hasil numerik final tereksekusi di snapshot. 

Tren umum yang diharapkan:
- 1 epoch: training loss lebih tinggi, EM/F1 sedikit lebih rendah.
- 2 epoch: titik sweet-spot antara kualitas dan waktu. 
- 3 epoch: potensi peningkatan kecil atau overfitting tergantung subset dan regularisasi. 
