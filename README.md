# Data and Code Release for "On the Potential of Lexico-logical Alignments for Semantic Parsing to SQL Queries"

## What are included

- Squall dataset
- Seq2seq-based models with attention and copy mechanisms (LSTM/BERT encoder)
- Supervised attention and column prediction using manually-annotated alignments

## Licenses

Our code has MIT license. The evalutor contains modified code of [mistic-sql-parser](https://github.com/mistic100/sql-parser) by Damien "Mistic" Sorel and Andrew Kent.

Th Squall dataset has CC BY-SA 4.0 license and is build upon [WikiTableQuestions] (https://github.com/ppasupat/WikiTableQuestions) by Panupong Pasupat and Percy Liang.

## Requirements

- python 3.x
- nodejs (for evaluator, details below)

## Setting Up

After `cd scripts`, run `python make_splits.py` to generate the train-dev splits used in our experiments; `./download_corenlp.sh` will download and unzip the corresponding CoreNLP version.

To set up the evaluator, `cd eval`, and then run `npm install file:sql-parser` and `npm install express`.

To set up the python dependencies, run `pip install -r requirements.txt`.

## Model Training and Testing

Make sure the evaluator service is running before performing any model training or testing. To do so, `cd eval` and run `node evaluator.js`. This will spawn a local service (default port 3000) that allows communication with the python model code to convert the (slightly) underspecified SQL queries into SQL queries fully-executable on our pre-processed databases.

Next, `cd model` and then run `python main.py` to train a baseline model with LSTM encoder, additional options to include our model variations:
- `--bert` for BERT encoder
- `--enc-loss` for encoder supervised attention
- `--dec-loss` for decoder supervised attention
- `--aux-col` for supervised column prediction

Once the model is trained, run `python main.py --test` to make predictions on the WTQ test set.

See `model/main.py` for command-line arguments to specify training file, dev file, test file, model saving location, etc.

## Squall Dataset Format

The dataset is located at `data/squall.json` as a single JSON file. The file is a list of dictionaries, each corresponding to one annotated data instance with the following fields:
- `nt`: question ID
- `tbl`: Table ID
- `columns`: a list of processed table columns with the format of \[raw header text, tokenized header text, available column suffixes (ways to interpret this column beyond raw texts), column data type\]
- `nl`: tokenized English question
- `tgt`: target execution result
- `nl_pos`: automatically-analyzed POS tags for `nl`
- `nl_ner`: automatically-analyzed NER tags for `nl`
- `nl_ralign`: automatically-generated field that includes information about what type of SQL fragments each question token aligns to, used in the auxiliary task of column prediction.
- `nl_incolumns`: Boolean values of whether the token matches any of the column tokens
- `nl_incells`: Boolean values of whether the token matches any of the table cells
- `columns_innl`: Boolean values of whether the column header appears in the question
- `sql`: tokenized SQL queries, each token has the format of \[SQL type, value, span indices\], SQL type is one among `Keyword`, `Column`, `Literal.Number`, `Literal.String`. If the token is a literal, then the span indices include the beginning and end indices to extract the literal from `nl`.
- `align`: Manual alignment between `nl` and `sql`. Each record in this list is in the format of \[indices into `nl`, indices into `sql`\]

## Release History

- `0.1.0` (2020-10-20): initial release

## Reference

If you make use of our code or data for research purposes, we'll appreciate your citing the following:

```
@inproceedings{Shi:Zhao:Boyd-Graber:Daume-III:Lee-2020,
	Title = {On the Potential of Lexico-logical Alignments for Semantic Parsing to {SQL} Queries},
	Author = {Tianze Shi and Chen Zhao and Jordan Boyd-Graber and Hal {Daum\'{e} III} and Lillian Lee},
	Booktitle = {Findings of EMNLP},
	Year = {2020},
}
```
