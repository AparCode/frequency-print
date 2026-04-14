const uploadInput = document.getElementById('audio');
const audioPlayer = document.getElementById('audioPlayer');
const predictBtn = document.getElementById('predictBtn');
const predictionResult = document.getElementById('predictionResult');

let currentObjectUrl = null;
let selectedAudioFile = null;

function getSelectedAudioFile() {
    return selectedAudioFile;
}

async function getSelectedAudioBytes() {
    if (!selectedAudioFile) {
        return null;
    }

    return await selectedAudioFile.arrayBuffer();
}

async function sendToBackEnd(file) {
    const data = new FormData();
    data.append('audio', file);
    const response = await fetch('/predict', {
        method: "POST",
        body: data
    });

    if (!response.ok) {
        throw new Error('Prediction request failed');
    }

    return await response.json();
}

window.getSelectedAudioFile = getSelectedAudioFile;
window.getSelectedAudioBytes = getSelectedAudioBytes;

uploadInput.addEventListener('change', function(event){
    const file = event.target.files[0];
    if(file){
        selectedAudioFile = file;

        if (currentObjectUrl) {
            URL.revokeObjectURL(currentObjectUrl);
        }

        const url = URL.createObjectURL(file);
        currentObjectUrl = url;
        audioPlayer.src = url;
        audioPlayer.load();
        predictionResult.textContent = '';
    }
});

predictBtn.addEventListener('click', async function() {
    const file = getSelectedAudioFile();
    if (!file) {
        predictionResult.textContent = 'Please upload an audio file first.';
        return;
    }

    predictionResult.textContent = 'Running prediction...';

    try {
        const data = await sendToBackEnd(file);
        const label = data.label || data.prediction || 'unknown';
        const hasConfidence = data.confidence !== undefined && data.confidence !== null;
        const confidence = hasConfidence
            ? ' (' + (Number(data.confidence) * 100).toFixed(2) + '%)'
            : '';

        predictionResult.textContent = 'Prediction: ' + label + confidence;
    } catch (error) {
        predictionResult.textContent = 'Error: ' + error.message;
    }
});