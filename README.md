# LLaVA iOS - Run LLaVA vision model on your phone! Read the [blog post](https://prashanth.world/llava-on-ios/)
This app lets you run the [LLaVA](https://github.com/haotian-liu/LLaVA/) multi modal LLM on your iPhone. I've only tested it with iPhone 15 pro, and it will crash fairly often.

This project is built using [llama.cpp](https://github.com/ggerganov/llama.cpp). All the LLaVA inference code is ripped from the llava example in there. even the UI bits.

Very much a work in progress.

## May 9th, 2024

The last commit now uses the models I re-trained using the training scripts in the LLava v1.5 github, using [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) as the base model, and inference has been improved to work much better! Here's an example (forgive the picture of my dirty dishes 😅):

![IMG_4101](https://github.com/prashanthsadasivan/llava-ios/assets/716375/1fb19b16-bb5a-454d-9578-8944f8073e38)

Inference is fast (though I'm not sure I trust the timings it displays), but it takes a little while to heat up, mainly i think because converting the image to clip embeddings takes a bit of time.
