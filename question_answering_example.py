"""
Zero-Shot Question Answering Example

This script demonstrates how to perform zero-shot question answering using
pre-trained models. The model can answer questions based on any context
without any task-specific training!
"""

from transformers import pipeline

def main():
    print("=" * 60)
    print("Zero-Shot Question Answering Example")
    print("=" * 60)
    print()
    
    # Initialize the question-answering pipeline
    print("Loading question-answering model...")
    qa_pipeline = pipeline("question-answering", 
                          model="distilbert-base-cased-distilled-squad")
    print("Model loaded successfully!")
    print()
    
    # Example contexts and questions
    examples = [
        {
            "context": """
            The Transformer architecture was introduced in the paper 'Attention is All You Need' 
            by Vaswani et al. in 2017. It revolutionized natural language processing by using 
            self-attention mechanisms instead of recurrent layers. This architecture has become 
            the foundation for models like BERT, GPT, and T5.
            """,
            "questions": [
                "When was the Transformer architecture introduced?",
                "Who introduced the Transformer?",
                "What mechanism does the Transformer use?",
                "What models are based on the Transformer architecture?"
            ]
        },
        {
            "context": """
            Zero-shot learning is a machine learning paradigm where a model can recognize and 
            classify objects or concepts it has never been explicitly trained on. This is 
            achieved by leveraging knowledge transfer from related tasks and semantic 
            relationships between labels. Popular zero-shot models include CLIP for vision 
            and BART for natural language processing.
            """,
            "questions": [
                "What is zero-shot learning?",
                "What are some popular zero-shot models?",
                "How does zero-shot learning work?"
            ]
        }
    ]
    
    # Process each example
    for i, example in enumerate(examples, 1):
        print(f"Context {i}:")
        print("-" * 60)
        print(example["context"].strip())
        print()
        
        for question in example["questions"]:
            result = qa_pipeline(question=question, context=example["context"])
            
            print(f"Q: {question}")
            print(f"A: {result['answer']}")
            print(f"Confidence: {result['score']:.2%}")
            print()
        
        print()
    
    # Interactive mode
    print("=" * 60)
    print("Try your own question!")
    print("=" * 60)
    print()
    
    custom_context = input("Enter context (or press Enter to use default): ").strip()
    
    if not custom_context:
        custom_context = """
        Python is a high-level, interpreted programming language created by Guido van Rossum 
        and first released in 1991. It emphasizes code readability and allows programmers to 
        express concepts in fewer lines of code than languages like C++ or Java. Python is 
        widely used in web development, data science, artificial intelligence, and automation.
        """
    
    print()
    print("Context:", custom_context.strip())
    print()
    
    while True:
        question = input("Enter your question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q', '']:
            break
        
        try:
            result = qa_pipeline(question=question, context=custom_context)
            print()
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['score']:.2%}")
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()

if __name__ == "__main__":
    main()
