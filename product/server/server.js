const express = require('express');
const axios = require('axios');
const app = express();
const path = require('path');
const port = 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'frontend')));
const flaskBaseUrl = 'http://ett-service:5000'; // Docker Compose service name and port

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

app.post('/ett', async (req, res) => {
  const response = await axios.post(flaskBaseUrl + '/' + req.body.destination , req.body,{ headers: {
    "Content-type": "application/json",}}
  );
  res.json(response.data);
});


// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});