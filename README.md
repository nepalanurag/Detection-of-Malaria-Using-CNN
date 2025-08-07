# Detection of Malaria Using CNN

This project leverages Convolutional Neural Networks (CNN) to automate the detection of malaria parasites in blood smear images. The goal is to assist healthcare professionals in rapid and accurate diagnosis, reducing manual workload and improving detection rates. The repository includes data preprocessing, model training, evaluation, prediction functionalities, visualizations, and a comprehensive project report.

## Background

Malaria is a life-threatening disease caused by Plasmodium parasites, transmitted through the bites of infected mosquitoes. Early and accurate detection is crucial for effective treatment. Manual examination of blood smears is time-consuming and prone to human error. Deep learning, especially CNNs, has shown promise in automating image-based diagnosis tasks.

## Project Structure

- `Welcome.py`: Main entry point or introduction script.
- `malaria_cnn.h5`, `my_model.h5`: Pre-trained CNN model files.
- `data.csv`, `incidence-of-malaria-sdgs.csv`: Datasets used for analysis and training.
- `archive/`: Contains additional datasets and CSV files.
- `pages/`: Contains scripts and notebooks for data cleaning, prediction, progression analysis, visualization, and exported data.
- `notebooka4911bf708.ipynb`: Jupyter notebook for experiments and analysis.
- `requirements.txt`, `Pipfile`: Python dependencies.
- `reports.pdf`: Project report and documentation.
- Image files: Sample images for testing and demonstration.

## Dataset

The primary dataset consists of cell images labeled as infected or uninfected. Additional CSV files provide malaria incidence statistics and related data. Data sources include the World Health Organization and open-access malaria datasets.

## How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   or use Pipenv:

   ```bash
   pipenv install
   ```

2. **Run prediction or analysis:**

   - Use scripts in the `pages/` directory, such as `Predict.py`, `Progression.py`, or `Visualization.py`.
   - Example:
     ```bash
     python pages/Predict.py
     ```

3. **Jupyter Notebook:**
   - Open `notebooka4911bf708.ipynb` or `pages/data_cleaning.ipynb` for step-by-step analysis and model training.

## Model Architecture

The CNN model is designed to classify cell images as malaria-infected or uninfected. The architecture typically includes:

- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Dense (fully connected) layers for classification
- Dropout layers to prevent overfitting

The model is trained using Keras and TensorFlow, with performance evaluated on validation and test sets. Pre-trained models (`malaria_cnn.h5`, `my_model.h5`) are provided for quick inference.

## Features

- Data cleaning and preprocessing
- CNN model training and evaluation
- Prediction on new images
- Visualization of results
- Comprehensive project report (`reports.pdf`)

## Usage Examples

**Predict malaria from an image:**

```python
from pages.Predict import predict_image
result = predict_image('sample.png')
print('Prediction:', result)
```

**Visualize malaria incidence:**

```bash
python pages/Visualization.py
```

## Requirements

- Python 3.7+
- TensorFlow, Keras, NumPy, Pandas, Matplotlib (see `requirements.txt`)

## Contributing

Contributions are welcome! To contribute:

- Fork the repository
- Create a new branch for your feature or bugfix
- Submit a pull request with a clear description of your changes

## References

- [World Health Organization Malaria Datasets](https://www.who.int/data/gho/data/themes/malaria)
- Project report: See `reports.pdf`

## Contact

For questions or feedback, please open an issue or contact the repository maintainer.

## License

This project is for educational and research purposes only. Please cite appropriately if you use this work in your research.
