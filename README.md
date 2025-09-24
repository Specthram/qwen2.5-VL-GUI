# qwen2.5-VL-GUI
grado interface for image batch captioning

A very simple GUI interface to batch caption images.
a .txt is generated next to the image with the same name, in the input folder.

some simple tools are provided like single captioning/remove/lock/prompts save

you can set temperature (0.7 by default), and seed (-1 by default).

The model used here is "Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed", giving decent results.


# Installation

Before install, check the .bat and modify if needed. Note that cuda is set to 12.8 here.

## Windows
launch .bat
it gonna create a venv and install needed dependencies.

## Linux
Just open .bat and execute venv/pip commands to install dependencies.

Feel free to open merge request to add features.

<img width="1511" height="993" alt="image" src="https://github.com/user-attachments/assets/b8030af4-57d5-4ae7-b667-189ae7b5a44f" />