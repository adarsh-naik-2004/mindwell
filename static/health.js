document.addEventListener('DOMContentLoaded', function() {
    const recordButton = document.getElementById('recordButton');
    const audioPlayback = document.getElementById('audioPlayback');
    const assessmentForm = document.getElementById('assessmentForm');
    const audioVisualizer = document.getElementById('audioVisualizer');

    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let animationId;

    // Create audio bars for visualizer
    for (let i = 0; i < 20; i++) {
        const bar = document.createElement('div');
        bar.className = 'audio-bar';
        audioVisualizer.appendChild(bar);
    }

    function updateAudioVisualizer() {
        const bars = audioVisualizer.children;
        for (let i = 0; i < bars.length; i++) {
            const height = Math.random() * 100;
            bars[i].style.height = `${height}%`;
        }
        if (isRecording) {
            animationId = requestAnimationFrame(updateAudioVisualizer);
        }
    }

    function toggleRecording() {
        if (isRecording) {
            mediaRecorder.stop();
            recordButton.querySelector('.btn-text').textContent = 'Start Recording';
            recordButton.classList.remove('recording');
            isRecording = false;
            cancelAnimationFrame(animationId);
        } else {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.start();

                    mediaRecorder.addEventListener("dataavailable", event => {
                        audioChunks.push(event.data);
                    });

                    mediaRecorder.addEventListener("stop", () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        audioPlayback.src = audioUrl;
                        audioPlayback.style.display = 'block';
                    });

                    recordButton.querySelector('.btn-text').textContent = 'Stop Recording';
                    recordButton.classList.add('recording');
                    isRecording = true;
                    updateAudioVisualizer();
                });
        }
    }

    recordButton.addEventListener('click', toggleRecording);

    assessmentForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const feelingText = document.getElementById('feelingText').value;
        console.log('Text input:', feelingText);
        console.log('Audio recorded:', audioPlayback.src);
        
        // Animate the submit button
        const submitBtn = this.querySelector('.submit-btn');
        submitBtn.classList.add('submitted');
        submitBtn.innerHTML = '<span><b>Submitted!</b></span>';
        
        // Redirect to result.html after a short delay
        setTimeout(() => {
            window.location.href = 'result.html';
        }, 1000);
    });

    // Add subtle animation to the form on load
    assessmentForm.style.opacity = '0';
    assessmentForm.style.transform = 'translateY(20px)';
    setTimeout(() => {
        assessmentForm.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        assessmentForm.style.opacity = '1';
        assessmentForm.style.transform = 'translateY(0)';
    }, 100);
});