document.getElementById("playButton").onclick = function() {
    var modal = document.getElementById("videoModal");
    var video = document.getElementById("popupVideo");
    modal.style.display = "block";
    video.play();
};

document.getElementById("closeButton").onclick = function() {
    var modal = document.getElementById("videoModal");
    var video = document.getElementById("popupVideo");
    modal.style.display = "none";
    video.pause();
};
