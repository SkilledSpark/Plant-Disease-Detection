require('dotenv').config();
const express = require("express");
const app = express();
const PORT = 8080;

const tf = require('@tensorflow/tfjs');
const multer = require('multer');

// Load TensorFlow model
async function loadModel() {
    try {
        const model = await tf.loadLayersModel("C:/Users/Lakshya Singh/Documents/GitHub/PlantVillage-Dataset/raw/segmented/final/modelextra.keras");
        return model;
    } catch (error) {
        console.error('Error loading model:', error);
        throw new Error('Error loading model');
    }
}

// To parse json by default
app.use(express.json());

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Internal Server Error');
});

// Serve the index.html page
app.use(express.static('public'));

// Routes
app.listen(
    PORT,
    () => console.log(`Server running on http://localhost:${PORT}`)
);

app.get('/', (req, res) => {
    res.send('Main Page!');
});

app.post('/feedback', (req, res) => {
    const { name, email, message } = req.body;
    if (name && email && message) {
        res.status(200).send({
            message: "Thank you for your feedback!"
        });
    } else {
        res.status(400).send({
            message: "Please try again!"
        });
    }
});

// Configure multer for file uploads
const upload = multer({ dest: 'uploads/' });

app.post('/predict', upload.single('image'), async (req, res) => {
    const { file } = req;
    if (file) {
        try {
            // Load TensorFlow model
            const model = await loadModel();

            // Perform model prediction directly on the image data
            const prediction = model.predict(file.path);

            // Send prediction response
            res.status(200).send({
                message: "Prediction successful",
                prediction: prediction
            });
        } catch (error) {
            console.error('Error predicting image:', error);
            res.status(500).send({
                message: "Error predicting image"
            });
        }
    } else {
        res.status(400).send({
            message: "Please provide an image"
        });
    }
});
