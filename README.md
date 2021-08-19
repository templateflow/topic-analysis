# topic-analysis

Materials corresponding to the templateflow paper - https://doi.org/10.1101/2021.02.10.430678

## Installation

First, install the necessary packages:

```Bash
pip install -r requirements.txt
```

or

```Bash
conda install -c conda-forge nltk scipy wordcloud
```

Once the dependencies are installed, download some NLTK data:

```Bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```
