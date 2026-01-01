# English to Urdu Neural Machine Translation

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat&logo=keras)

##  Project Overview
This project implements a **Sequence-to-Sequence (Seq2Seq)** model for Neural Machine Translation (NMT), specifically translating **English sentences to Urdu**. The model is built using **Recurrent Neural Networks (SimpleRNN)** within the TensorFlow/Keras framework.

The goal of this project is to demonstrate the fundamental architecture behind language translation tasks, including text tokenization, padding, and the Encoder-Decoder architecture.

##  Model Architecture
The system uses a classic **Encoder-Decoder** architecture:

1.  **Encoder:** * Takes English sentences as input.
    * Uses an **Embedding Layer** to convert words into dense vectors.
    * Uses a **SimpleRNN layer** (512 units) to process the sequence and generate a context vector (state).
2.  **Decoder:**
    * Takes Urdu sentences as input (teacher forcing during training).
    * Uses the context vector from the encoder as its initial state.
    * Uses a **SimpleRNN layer** to predict the next word in the sequence.
3.  **Output:** A Dense layer with Softmax activation outputs the probability distribution over the Urdu vocabulary.

##  Dataset
The model works on a custom curated dataset containing parallel English-Urdu sentence pairs. 
* **Input:** English text (e.g., "How are you?")
* **Target:** Urdu text (e.g., "آپ کیسے ہیں؟")
* **Preprocessing:** Tokenization and padding were applied to handle variable sequence lengths.

## Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Natural Language Processing:** Tokenizer, Pad Sequences
* **Visualization:** Matplotlib (for training loss)

##  Key Improvements
Throughout the development, the model was optimized by:
* Increasing the dataset size manually to improve generalization.
* Scaling the RNN units from **256 to 512** to increase the model's capacity to learn complex patterns.

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PyPro2024/RNN-Eng-To-Urdu-Language-Translator.git]
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy matplotlib
    ```
3.  **Run the Notebook:**
    Open `English_to_Urdu_Translation_Using_RNN.ipynb` in Jupyter Notebook or Google Colab to train the model and test translations.

## 🔮 Future Scope
* Upgrade from SimpleRNN to **LSTM** or **GRU** to handle longer dependencies.
* Implement an **Attention Mechanism** for better translation accuracy on long sentences.
* Train on a larger corpus (e.g., Ankur Corpus).

---
*If you find this project interesting, feel free to ⭐ the repo!*
