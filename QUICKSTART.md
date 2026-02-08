# Quick Start Guide

Get started with zero-shot learning in minutes!

## Step 1: Clone the Repository

```bash
git clone https://github.com/isaid22/Zero-Shot.git
cd Zero-Shot
```

## Step 2: Set Up Python Environment

It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first time you run an example, the model will be downloaded automatically. This may take a few minutes depending on your internet connection.

## Step 4: Run Your First Example

### Text Classification

```bash
python text_classification_example.py
```

This will:
1. Load the zero-shot classification model
2. Classify several example texts
3. Allow you to try your own text interactively

**Expected output**:
```
Text 1: The Golden State Warriors won the NBA championship last night.
Predicted category: sports
Confidence: 98.5%
```

### Image Classification

```bash
python image_classification_example.py
```

This will classify sample images into categories using natural language.

**Note**: This example downloads images from URLs. Make sure you have an internet connection.

### Question Answering

```bash
python question_answering_example.py
```

This will answer questions based on provided contexts and let you try your own.

**Expected output**:
```
Q: When was the Transformer architecture introduced?
A: 2017
Confidence: 95.3%
```

## Troubleshooting

### Issue: "No module named 'transformers'"
**Solution**: Make sure you installed the requirements: `pip install -r requirements.txt`

### Issue: Model download is slow
**Solution**: This is normal for the first run. Models are cached locally and won't be downloaded again.

### Issue: Out of memory error
**Solution**: These models can be large. Try:
- Close other applications
- Use a machine with more RAM
- Try smaller model variants

### Issue: ImportError for torch
**Solution**: Install PyTorch manually:
```bash
# CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU version (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps

1. **Customize examples**: Edit the Python files to use your own data
2. **Read documentation**: Check `EXAMPLES.md` for detailed information
3. **Experiment**: Try different models and parameters
4. **Build something**: Use these examples as a foundation for your project

## Need Help?

- Check the `EXAMPLES.md` file for detailed documentation
- Review the code comments in each example file
- Visit [Hugging Face documentation](https://huggingface.co/docs/transformers)
- Open an issue on GitHub

Happy zero-shot learning! 🚀
