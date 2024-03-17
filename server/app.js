import express from 'express';
import cors from 'cors';
import cookieParser from 'cookie-parser';
import { limit } from './constants.js';
import { createServer } from 'http';
import socketHandler from './socket/socketHandler.js';
import fs from 'fs';
import { spawn } from "child_process";

const app = express();
const httpServer = createServer( app );

app.use( express.json({
    limit,
    extended: true
}) )

app.use( express.urlencoded( 
    {
        extended: true,
        limit
    }
) );

app.use( cors({
    origin: "http://localhost:5173",
    credentials: true,
}) );

app.use( cookieParser() );

// Endpoint to receive image data from frontend
app.post('/uploadImage', (req, res) => {
    const { imageData } = req.body;
  
    // Decode base64 image data
    const base64Data = imageData.replace(/^data:image\/jpeg;base64,/, '');
  
    // Save the image to a file (optional, you can process it directly)
    fs.writeFile('image.jpg', base64Data, 'base64', (err) => {
      if (err) {
        console.error('Error saving image:', err);
        res.status(500).json({ error: 'Error saving image' });
      } else {

        const pythonScript = '../Predict.py';
        const pythonArgs = [base64Data];

        const pythonProcess = spawn('python', [pythonScript, ...pythonArgs]);

        pythonProcess.stdout.on('data', (data) => {
            console.log(`Python script stdout: ${data}`);
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Python script stderr: ${data}`);
        });

        pythonProcess.on('exit', (code) => {
            console.log(`Python script exited with code ${code}`);
        })
        console.log('Image saved successfully');

        fs.unlinkSync('image.jpg');

        res.status(200).json({ message: 'Image received and saved' });
      }
    });
  });

import userRoutes from "./routes/user.routes.js";
app.use("/api/v1/users", userRoutes);

import chatRoutes from "./routes/chat.routes.js";
app.use("/api/v1/chats", chatRoutes);

import messageRoutes from "./routes/message.routes.js"
app.use("/api/v1/messages", messageRoutes)

socketHandler( httpServer );

export { httpServer };