# Short Answer Questions

## Q1: Primary Differences Between TensorFlow and PyTorch
**TensorFlow**:
- Uses static computation graphs (by default, though Eager Execution allows dynamic graphs).
- Optimized for production deployment, with tools like TensorFlow Serving and TensorFlow Lite.
- Strong support for distributed training and large-scale systems.
- Steeper learning curve due to its comprehensive ecosystem.

**PyTorch**:
- Uses dynamic computation graphs, enabling flexible model building and debugging, ideal for research.
- Simpler, more intuitive API, preferred by beginners and researchers.
- Strong community support for academic prototyping but less focus on production tools compared to TensorFlow.

**When to Choose**:
- Choose **TensorFlow** for production-grade applications (e.g., deploying a model for SDG 13 emissions prediction on a server) or when distributed training is needed.
- Choose **PyTorch** for research, rapid prototyping, or when flexibility in model design is critical (e.g., experimenting with neural networks for climate data analysis).

## Q2: Two Use Cases for Jupyter Notebooks in AI Development
1. **Prototyping Models**: Jupyter Notebooks allow interactive coding and visualization, ideal for testing ML models like the SDG 13 carbon emissions predictor. Developers can preprocess data, train models (e.g., using Scikit-learn), and visualize results (e.g., scatter plots) in one environment.
2. **Data Exploration and Teaching**: Notebooks support exploratory data analysis (EDA) by combining code, markdown, and visualizations. For SDG projects, researchers can share notebooks to demonstrate how datasets (e.g., World Bank climate data) correlate with outcomes, making it a great tool for collaboration and education.

## Q3: How spaCy Enhances NLP Tasks Compared to Basic Python String Operations
- **Efficiency**: spaCy uses optimized Cython code for fast tokenization, part-of-speech tagging, and named entity recognition (NER), unlike slow, manual string operations in Python.
- **Pre-trained Models**: spaCy provides pre-trained NLP models for tasks like NER (e.g., extracting product names from Amazon reviews), while string operations require custom regex or splitting logic.
- **Context Awareness**: spaCy understands linguistic context (e.g., sentence structure, dependencies), enabling accurate entity extraction and sentiment analysis, whereas string operations are limited to pattern matching.
- **Scalability**: spaCy handles large datasets efficiently, crucial for SDG 5 (Gender Equality) tasks like analyzing social media for gender bias, compared to brittle, error-prone string methods.