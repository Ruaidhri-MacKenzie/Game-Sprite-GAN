const express = require("express");
const router = require("./router.js");

const PORT = process.env.PORT || 2000;
const PUBLIC_PATH = __dirname + "/public";

const app = express();

// Make JSON sent in the request body available as req.body
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Serve static files
app.use(express.static(PUBLIC_PATH));

// Routes
router.get("/", (req, res) => res.sendFile(PUBLIC_PATH + "/index.html"));
app.use("/model", router);

// Page not found - standard redirect
app.use((req, res) => res.redirect("/"));

// Start server listening for requests
app.listen(PORT, () => console.log(`Server listening on port ${PORT}...`));
