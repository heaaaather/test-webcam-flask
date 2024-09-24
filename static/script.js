// create a socket object to capture the video stream from the user's camera
var socket = io.connect(
    window.location.protocol + "//" + document.domain + ":" + location.port
  );
  socket.on("connect", function () {
    console.log("Connected...!", socket.connected);
  });

  // set up canvas to capture the video stream from the user's camera
  var canvas = document.getElementById("canvas");
  var context = canvas.getContext("2d");
  const video = document.querySelector("#videoElement");
  
  video.width = 400;
  video.height = 300;
  
  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia({
        video: true,
      })
      .then(function (stream) {
        video.srcObject = stream;
        video.play();
      })
      .catch(function (err0r) {});
  }

  // to capture the video stream from the user's camera at a certain frame rate and;
  // send it to the server for processing
  const FPS = 10;
  setInterval(() => {
    width = video.width;
    height = video.height;
    context.drawImage(video, 0, 0, width, height);
    var data = canvas.toDataURL("image/jpeg", 0.5);
    context.clearRect(0, 0, width, height);
    socket.emit("image", data);
  }, 1000 / FPS);
  
  socket.on("processed_image", function (image) {
    photo.setAttribute("src", image);
  });