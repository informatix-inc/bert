# Japanese RoBERTa
This repository provides snippets to use RoBERTa pre-trained on Japanese corpus.
Our dataset consists of Japanese Wikipedia and web-scrolled articles, 25GB in total.
The released model is built based on that from [HuggingFace](https://huggingface.co/docs/transformers/index).

## Details

### Preprocessing
We used [Juman++](https://github.com/ku-nlp/jumanpp) (version 2.0.0-rc3) as a morphological analyzer, 
and also applied WordPiece embedding ([subword-nmt](https://github.com/rsennrich/subword-nmt)) to split each word into word pieces.

### Model
Configurations of our model are following.
Please refer to [HuggingFace page](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaConfig) for definitions of each parameter.
~~~
{
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 3,
  "classifier_dropout": null,
  "eos_token_id": 0,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 515,
  "max_seq_length": 512,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 2,
  "position_embedding_type": "absolute",
  "transformers_version": "4.16.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 32005
}
~~~
### Training
We trained our model in the same way as RoBERTa. We optimized our model using the masked language modeling (MLM) objective.
The accuracy of the MLM is 72.0%.

## How to use our model
1. Install Juman++ following the instructions from [this repository](https://github.com/ku-nlp/jumanpp).
2. Install PyTorch following the instructions from [this page](https://pytorch.org/get-started/locally/#start-locally).
3. Install all of the necessary python packages.
~~~
pip install -r requirements.txt
~~~
4. Download weight files and configuration files.

| Files | Description |
| --- | --- |
| [roberta.pth](https://ifx-bert.s3.ap-northeast-1.amazonaws.com/files/roberta.pth) | Trained weights of RobertaModel from HuggingFace |
| [linear_word.pth](https://ifx-bert.s3.ap-northeast-1.amazonaws.com/files/linear_word.pth) | Trained weights of linear layer for MLM |
| [roberta_config.json](https://ifx-bert.s3.ap-northeast-1.amazonaws.com/files/roberta_config.json) | Configurations for RobertaModel |
| [bpe.txt](https://ifx-bert.s3.ap-northeast-1.amazonaws.com/files/bpe.txt) | Rules for splitting each word into word pieces |
| [bpe_dict.csv](https://ifx-bert.s3.ap-northeast-1.amazonaws.com/files/bpe_dict.csv) | Dictionary of word pieces |

5. Load weights and configurations.
~~~
    # Paths to each file
    bpe_file = <Path to the file (bpe.txt) which defined word pieces>
    count_file = <Path to the file (bpe_dict.csv) which defines ids for word pieces>
    roberta_config_path = <Path to the file (roberta_config.json) which defines configurations of RobertaModel>
    juman_config_path = <Path to config file for juman>
    
    roberta_weight_path = <Path to the weight file (roberta.pth) of RobertaModel>
    linear_weight_path = <Path to the weight file (linear_word.pth) of final linear layer for MLM>
    
    # load tokenizer
    processor = TextProcessor(bpe_file=bpe_file, count_file=count_file)

    # load pretrained roberta model
    with open(roberta_config_path, "r") as f:
        config_dict = json.load(f)
    config_bert = RobertaConfig().from_dict(config_dict)
    roberta = RobertaModel(config=config_roberta)
    roberta.load_state_dict(torch.load(roberta_weight_path, map_location=device))

    # load pretained decoder
    ifxroberta = IFXRoberta(roberta)
    ifxroberta.linear_word.load_state_dict(torch.load(linear_weight_path, map_location=device))
~~~
6. Encode inputs.
~~~
    # infer
    inp_text = "コンピュータ技術およびその関連技術に対する研究開発に努め、自己研鑽に励み、専門的な技術集団を形成する。"
    bpe_text = processor.encode(inp_text)
    with torch.no_grad():
        inp_tensor = torch.LongTensor([bpe_text])
        inp_tensor.to(device)
        out_tensor = ifxroberta(inp_tensor)
~~~
7. Decode model outputs.
~~~
    # decoding output
    out_text_code = torch.max(out_tensor, dim=-1, keepdim=True)[1][0]
    ids = out_text_code.squeeze(-1).cpu().numpy().tolist()
    ignore_idx = np.where(np.array(bpe_ids) < 5)[0]
    out_text = processor.decode(ids, ignore_idx=ignore_idx)
~~~
8. `main.py` helps you understand how to use our model.

## License
[The Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)
