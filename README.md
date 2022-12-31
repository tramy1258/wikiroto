This dataset contains critic reviews as user profiles and movie infoboxes as data-to-text input with a goal to study the personalization of movie descriptions.
## Additional data and code source
This repository contains data extracted from [PENS](https://msnews.github.io/pens.html) and [Kaggle Rotten Tomatoes Reviews](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) and a extended dataset built upon the Kaggle dataset by scraping Wikipedia for movie infoboxes and Rotten Tomatoes for critic reviews.

For evaluation, [this](https://github.com/KaijuML/parent) implementation is used to compute the PARENT score, make a clone of the repository under `src/`.

`git clone https://github.com/KaijuML/parent`
## To train T5 models on generating movie description from movie infobox
From `src/`, to train any size of T5 model

`python dtt_T5.py t5_model_size`

-- `t5_model_size` could be, for example, `t5-small`, `t5-base`, etc.

## To tune prompts using trained T5 models
- Download `news.tsv` from PENS, add it under `data/pens/`. (Only for additional tuning)

`python perso_prompt_tuning.py t5_model_size prompt_length batch_training_size model_path prompt_path` 

-- `prompt_length` and `batch_training_size` are integers.

-- `model_path` and `prompt_path` could be `.` if train from scratch or path to an `.pth` file.

## To scrape reviews
- Download `rotten_tomatoes_critic_reviews.csv` and `rotten_tomatoes_movies.csv` from Kaggle, add them under `data/`. (Only for additional scraping)

`python scraping_reviews.py 0`
