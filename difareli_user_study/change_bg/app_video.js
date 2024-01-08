const express = require('express');
const fs = require('fs');
const app = express();
let root = './'
app.use(express.json())
app.use(express.static(root))

const port = 3000;

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/change_bg_with_video.html');
});

app.post("/submit", function(req, res) {
  var data = req.body;
  var submit_t = Date.now();
  console.log(`[#] Data received from ${submit_t} : `);
  output = {}
  output[submit_t] = data
  console.log(output)
  fs.writeFile(`./saved_file/${submit_t}.json`, JSON.stringify(output), 'utf8', function (err) {
    if (err) throw err;
    console.log('The file has been saved!');
  });
  res.json({ message: "Data received and saved successfully" });
});

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`);
});