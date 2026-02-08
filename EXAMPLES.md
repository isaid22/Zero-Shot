# Examples Documentation

This directory contains detailed documentation for each zero-shot learning example.

## Text Classification

### Overview
The text classification example demonstrates how to classify text into arbitrary categories without training data. It uses the BART model fine-tuned on MNLI (Multi-Genre Natural Language Inference) for zero-shot classification.

### How it works
1. The model was pre-trained on natural language inference tasks
2. Classification is framed as an entailment problem
3. For each category, the model checks if the text entails the hypothesis "This text is about [category]"
4. The category with the highest entailment score wins

### Use Cases
- Sentiment analysis
- Topic classification
- Intent detection
- Content moderation
- Document categorization

### Customization
You can easily customize:
- **Categories**: Use any categories you want (e.g., "positive", "negative", "neutral")
- **Multi-label**: Set `multi_label=True` to allow multiple categories per text
- **Model**: Try different models like "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

---

## Image Classification

### Overview
The image classification example uses CLIP (Contrastive Language-Image Pre-training) to classify images using natural language descriptions.

### How it works
1. CLIP was trained on 400 million image-text pairs from the internet
2. It learned to match images with their textual descriptions
3. For classification, it compares the image with text descriptions of each category
4. The most similar text description determines the category

### Use Cases
- Product categorization
- Content tagging
- Visual search
- Object detection
- Scene understanding

### Customization
You can:
- **Use local images**: Replace URLs with local file paths
- **Custom descriptions**: Use detailed descriptions (e.g., "a photo of a happy dog playing in a park")
- **Different models**: Try "openai/clip-vit-large-patch14" for better accuracy

---

## Question Answering

### Overview
The question answering example demonstrates extractive QA, where the model finds and extracts the answer from a given context.

### How it works
1. The model was pre-trained on question-answer pairs (SQuAD dataset)
2. It learns to identify the span of text that answers the question
3. No task-specific fine-tuning needed for new contexts
4. Returns answer with confidence score

### Use Cases
- Document QA systems
- Customer support automation
- Information extraction
- Reading comprehension
- FAQ systems

### Customization
You can:
- **Different contexts**: Provide any text as context
- **Multiple questions**: Ask multiple questions about the same context
- **Better models**: Try "deepset/roberta-base-squad2" for improved performance
- **Long contexts**: Use models designed for longer texts

---

## Tips for Best Results

### Text Classification
- Use clear, distinct categories
- Keep category names simple and descriptive
- Try different hypothesis templates if results aren't good
- For ambiguous cases, check all probability scores

### Image Classification
- Use descriptive text for categories
- Include relevant details in descriptions (e.g., "outdoor", "indoor")
- Higher resolution images generally work better
- Try both specific and general descriptions

### Question Answering
- Ensure the answer is actually in the context
- Ask specific, clear questions
- Keep context focused and relevant
- Questions should be answerable from the context alone

## Performance Considerations

- **First run**: Models are downloaded (can take several minutes)
- **Model size**: CLIP and BART models are large (several GB)
- **Inference speed**: CPU inference is slower; consider GPU for production
- **Caching**: Models are cached locally after first download

## Further Reading

- [Hugging Face Zero-Shot Classification](https://huggingface.co/tasks/zero-shot-classification)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Zero-Shot Learning Tutorial](https://huggingface.co/docs/transformers/tasks/zero_shot_classification)
