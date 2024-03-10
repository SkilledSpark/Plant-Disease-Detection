require('dotenv').config();
const sharp = require('sharp');
const express = require("express");
const app = express();
const PORT = 8080;

const tf = require('@tensorflow/tfjs');
const multer = require('multer');

// const mongoose = require("mongoose");
// // Connect to database
// mongoose.connect(process.env.DATABASE_URL, {
//     useNewUrlParser: true
// });
// const db = mongoose.connection;
// db.on("error", (error) => console.error(error));
// db.once("open", () => console.log("Connected to database"));


// Load TensorFlow model
async function loadModel() 
{
    try {
        const model = await tf.loadLayersModel('path_to_model');
        return model;
    } 
    catch (error) 
    {
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
    const {name, email, message} = req.body;
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

            // Resize and normalize image
            const processedImage = await processImage(file.path);

            // Perform model prediction
            const prediction = model.predict(processedImage);

            // Send prediction response
            res.status(200).send({
                message: "Prediction successful",
                prediction: prediction
            });
        } catch (error) {
            console.error('Error processing or predicting image:', error);
            res.status(500).send({
                message: "Error processing or predicting image"
            });
        }
    } else {
        res.status(400).send({
            message: "Please provide an image"
        });
    }
});

async function processImage(imageData) 
{
    try 
    {
        // Resize image to 256x256 pixels
        const resizedImage = await sharp(imageData)
            .resize(256, 256)
            .toBuffer();

        // Convert image to JPEG format
        const jpegImage = await sharp(resizedImage)
            .jpeg()
            .toBuffer();

        return jpegImage;
    } 
    catch (error) 
    {
        console.error('Error pre-processing image:', error);
        throw new Error('Error pre-processing image');
    }
}
