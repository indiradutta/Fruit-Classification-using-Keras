# Fruit Classification
<br>
Fruits are very common in today’s world – despite the abundance of fast food and refined sugars, fruits remain widely consumed foods. During production of fruits, it might be that they need to be sorted, to give just one example. Traditionally being performed mechanically, today, deep learning based techniques could augment or even take over this process.
<br>

## Dataset
<br>

The dataset used for this project is the <a href = "https://www.kaggle.com/moltean/fruits">Fruit Classification</a> dataset from Kaggle.
The Fruit Classification Dataset contains 6 classifications:

- Apple
- Avocado
- Banana
- Blueberry
- Cucumber
- Mango
<br>

The raw data has been manually updated in a format where there are 2 main folders namely: train and test, each containing the 6 classes.
## Steps Involved:

- Viewing classes in Directory

- Visualizing Images in Dataset from each class

- Data Configuration

- Data Preparation and Loading

    - Creating a Generator for Training Set
    - Creating a Generator for Testing Set
    
- Writing the labels into a text file 'Labels.txt 

- Model Architecture

- Model Compilation

- Training the Model (batch_size = 32, epochs = 4)

- Testing Predictions

- Deploying the Model as a Web Application using Streamlit
