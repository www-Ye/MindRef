# MindRef: Mimicking Human Memory for Hierarchical  Reference Retrieval with Fine-Grained Location Awareness

Code implementation of the ACL 2025 main paper [MindRef: Mimicking Human Memory for Hierarchical  Reference Retrieval with Fine-Grained Location Awareness](https://aclanthology.org/2025.acl-short.67/).

## Prerequisites

- Install or clone the [KILT](https://github.com/facebookresearch/KILT) repository.
- Install [SEAL](https://github.com/facebookresearch/SEAL).

1.  **Clone the MindRef repository:**
    ```bash
    git clone https://github.com/www-Ye/MindRef
    cd MindRef
    mkdir predictions
    cd ..
    ```

2.  **Create a Conda environment and install PyTorch:**
    ```bash
    conda create -n mindref python=3.9
    conda activate mindref
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

3.  **Install KILT:**
    ```bash
    git clone https://github.com/facebookresearch/KILT
    cd KILT
    pip install -e .
    cd .. # Return to the MindRef directory or your preferred working directory
    ```

4.  **Install SEAL:**
    ```bash
    git clone --recursive https://github.com/facebookresearch/SEAL.git
    cd SEAL
    env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh
    pip install -r requirements.txt
    pip install -r requirements_extra.txt
    pip install -e .
    ```

5.  **Apply patches to SEAL for compatibility with newer `transformers` versions:**
    *   **Edit `seal/beam_search.py`:**
        -   Change `from transformers.generation_utils import xxx` to `from transformers.generation.utils import xxx`
        -   Change `from transformers.generation_logits_process import TopKLogitsWarper` to `from transformers.utils.dummy_pt_objects import TopKLogitsWarper`

6.  **Install a compatible `transformers` version:**
    ```bash
    pip install transformers==4.28.0
    ```

## Dataset Download

Download the validation sets of the following datasets from the [KILT](https://github.com/facebookresearch/KILT) repository:
- NQ
- HotpotQA
- TriviaQA
- ELI5
- FEVER
- WoW

Follow the instructions in the KILT repository for downloading the specific validation sets.

## Generating Retrieval Documents

### Step 1: Download the Official KILT Splits

- Download [kilt_w100_title.tsv](http://dl.fbaipublicfiles.com/KILT/kilt_w100_title.tsv) and [mapping_KILT_title.p](http://dl.fbaipublicfiles.com/KILT/mapping_KILT_title.p).

### Step 2: Reconstruct Documents from Segmented Paragraphs

- Run the `passage2doc.sh` script:
  ```sh
  ./passage2doc.sh
  ```

### Step 3: Build Index

- Run the `build_index.sh` script:
  ```sh
  ./build_index.sh
  ```

## Running Generative Retrieval

- Execute the `run.sh` script to perform generative retrieval:
  ```sh
  ./run.sh
  ```
