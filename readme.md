Sheet Count Predictor

    This project is a web application that predicts the number of sheets in an image.
    The application uses a trained deep learning model to analyze the image and return
    the sheet count along with the edge count of the preprocessed image.

Files & Purpose

- uploads/: Directory where uploaded images are stored.
- templates/: Directory containing HTML templates.
- app.py: Flask application script.
- preprocess.py: Contains functions for preprocessing images.
- model.py: Contains functions to load the trained model and make predictions.
- train_model.py: Script to train the deep learning model (optional, not required if you already have `model.keras`).
- model.keras: Pretrained model file.


Installation

    1. Clone the repository:

       Terminal
           git clone https://github.com/OPanurag/ComputerHubCVTask.git
           cd ComputerHubCVTask

    2. Create a Virtual Environment

        Terminal
            python -m venv venv
            source venv/bin/activate   # On Windows use `venv\Scripts\activate`

    3. Install Required Packages
        Terminal
            pip install -r requirements.txt


Training the Model

    If you don't have a pretrained model (model.keras), you can train one using the train_model.py script.
    Ensure you have your dataset prepared.

    1. Prepare your dataset and place images and annotations.csv in the synthetic_data directory.

    2. Run the training script:

    Terminal
        python train_model.py

    The trained model will be saved as model.keras.

Running the Application

    1. Ensure you have the pretrained model file model.keras in the project root directory.

    2. Start the Flask server:

    Terminal
        python app.py

    3. Open your web browser and navigate to http://127.0.0.1:5000/.

Usage
    1. On the web page, click "Choose File" to select an image from your computer.
    2. Click "Upload" to upload the image.
    3. The application will display the predicted sheet count and edge count.