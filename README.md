# Caption Forge

A local based captioning tool with graphic interface (gradio), using Qwen2.5-VL models.

*__🚧 Early stage project – the code is messy, experimental and under heavy development.  
Don’t expect clean architecture (yet)!__*

## Features

- caption images
- caption first frame of videos
- batch captioning / single captioning
- caption live editing
- lock/secure captions to make new batch
- remove images
- model selection
- prompt customization
- load/unload model
- temperature / seed settings
- download a zip archive of your captioned dataset

## Supported Models Types

At the moment, there is two kind of models caption-forge is compatible.
- Qwen2.5-Vl
- Llava (joycaption)

you can edit the models.json file to add your own qwen2.5-llava models.

## What happens behind the scenes ?

the first time you caption, the model is downloaded from huggingface.
the pictures you upload are placed in an input folder. captions are saved in a .txt with the same name of your picture.

```
image1.jpg
image1.txt
```
You can drag and drop directly inside/outside from input folder.

A config folder is created, with `prompts.json` and `models.json` feel free to edit or modify for your needs.
the GUI allows you to add prompts but not remove them at the moment.


# Installation

This is currently made for cuda 12.8, you can modifiy as needed.

## Windows

run `run.bat`

## Linux

```bash
python -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python app.py
```

## Special notes

be careful with remove feature.

## Roadmap

The code is in a dirty state, just wanted it to work in a first time

Feel free to ask / make a PR to add something you need (you should ask before if if something I want in), or just for code cleaning.

The feature I see in the future :

- models in models folder, to avoid download models you already have
- better video captioning
- other models (qwen3, joy caption...)
- better GUI (with tabs etc)
- dataset switching (input/myDataset1, input/myDataset2 etc)
- multiple captioning : a model needs a specific captioning, sometimes booru tags, sometimes long/short captions style.
    ```
    image1.jpg
    image1.flux1.txt
    image1.wanvideo.txt
    image1.illustrious2.txt
    ```
- dataset deploying, creating symlinks in your ai_toolkit/kohya/whatever folder to avoid duplication
- dynamic dataset creation, using sqlite or something like that, to create and store theses datasets virtualy before deploy
- automatic image renaming - avoid the error of `image1.png` and `image1.jpg` overwriting same `image1.txt` file
- word finder/replacer

### Feel free to suggest a feature.

<img width="1249" height="1283" alt="image" src="https://github.com/user-attachments/assets/d24ef4b6-ef5d-4ef9-957f-58197cf34502" />


