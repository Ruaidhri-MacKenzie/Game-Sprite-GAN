const express = require("express");
const controller = require("./controller.js");

const router = express.Router();

router.post("/", controller.save);
router.get("/", controller.load);
router.get("/reset", controller.reset);
router.get("/train", controller.train);
router.get("/test", controller.test);

module.exports = router;
