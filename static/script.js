const videoFeed = document.getElementById('videoFeed');

function startCamera() {
    fetch('/start')
        .then(res => res.json())
        .then(data => {
            console.log(data);
            // Reload video feed
            videoFeed.src = '/video?' + new Date().getTime(); // bust cache
            videoFeed.style.display = 'block';
        });
}

function stopCamera() {
    fetch('/stop')
        .then(res => res.json())
        .then(data => {
            console.log(data);
            videoFeed.src = '';
            videoFeed.style.display = 'none';
        });
}

document.addEventListener('keydown', function (e) {
    if (e.code === 'Space') {
        startCamera();
    } else if (e.key.toLowerCase() === 'q') {
        stopCamera();
    }
});