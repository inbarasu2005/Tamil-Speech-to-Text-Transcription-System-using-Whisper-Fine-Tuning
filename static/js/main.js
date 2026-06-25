document.addEventListener("DOMContentLoaded", () => {
    // ----------------------------------------------------
    // State Variables
    // ----------------------------------------------------
    let mediaRecorder = null;
    let audioChunks = [];
    let recordStartTime = null;
    let timerInterval = null;
    let recordedBlob = null;
    let selectedFile = null;

    // Volume monitoring variables
    let audioContext = null;
    let analyser = null;
    let dataArray = null;
    let source = null;
    let volumeCheckInterval = null;

    // ----------------------------------------------------
    // DOM Elements
    // ----------------------------------------------------
    // Configurations
    const apiKeyInput = document.getElementById("apiKeyInput");
    const modelInput = document.getElementById("modelInput");

    // Live Recording Elements
    const recordBtn = document.getElementById("recordBtn");
    const recordIcon = document.getElementById("recordIcon");
    const pulseRing = document.getElementById("pulseRing");
    const recordStatus = document.getElementById("recordStatus");
    const recordDuration = document.getElementById("recordDuration");
    const recordPlaybackContainer = document.getElementById("recordPlaybackContainer");
    const audioPlayback = document.getElementById("audioPlayback");
    const transcribeRecordBtn = document.getElementById("transcribeRecordBtn");

    // File Upload Elements
    const dropZone = document.getElementById("dropZone");
    const audioFileInput = document.getElementById("audioFileInput");
    const fileUploadDetails = document.getElementById("fileUploadDetails");
    const fileNameText = document.getElementById("fileNameText");
    const fileSizeText = document.getElementById("fileSizeText");
    const filePlayback = document.getElementById("filePlayback");
    const removeFileBtn = document.getElementById("removeFileBtn");
    const transcribeUploadBtn = document.getElementById("transcribeUploadBtn");

    // Output Elements
    const transcriptionLoader = document.getElementById("transcriptionLoader");
    const resultContainer = document.getElementById("resultContainer");
    const transcriptionOutput = document.getElementById("transcriptionOutput");
    const statusBadge = document.getElementById("statusBadge");
    const copyTextBtn = document.getElementById("copyTextBtn");
    const downloadFileBtn = document.getElementById("downloadFileBtn");

    // Add media error logging for debugging
    if (audioPlayback) {
        audioPlayback.addEventListener("error", (e) => {
            console.error("audioPlayback media error:", audioPlayback.error);
        });
    }
    if (filePlayback) {
        filePlayback.addEventListener("error", (e) => {
            console.error("filePlayback media error:", filePlayback.error);
        });
    }


    // ----------------------------------------------------
    // Tab Change Reset Utility
    // ----------------------------------------------------
    const tabs = document.querySelectorAll('button[data-bs-toggle="tab"]');
    tabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', (e) => {
            // Reset results states
            resetResults();
            if (e.target.id === 'record-tab') {
                // Leaving Upload tab -> Reset Upload
                resetUploadState();
            } else {
                // Leaving Record tab -> Reset Record
                resetRecordState();
            }
        });
    });

    // ----------------------------------------------------
    // Audio Recording Functionality
    // ----------------------------------------------------
    recordBtn.addEventListener("click", async () => {
        if (!mediaRecorder || mediaRecorder.state === "inactive") {
            // Start recording
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioChunks = [];
                
                // Determine mime type support
                let options = { mimeType: 'audio/webm' };
                if (!MediaRecorder.isTypeSupported('audio/webm')) {
                    options = { mimeType: 'audio/ogg' };
                }
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options = {}; // Fallback to default browser implementation
                }

                mediaRecorder = new MediaRecorder(stream, options);
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = () => {
                    // Create audio blob safely
                    const mimeType = mediaRecorder.mimeType || (options && options.mimeType) || "";
                    recordedBlob = new Blob(audioChunks, mimeType ? { type: mimeType } : {});
                    const audioURL = URL.createObjectURL(recordedBlob);
                    audioPlayback.src = audioURL;
                    audioPlayback.load();

                    // Display details
                    recordStatus.innerText = "Recording finished! Check playback below.";
                    recordDuration.classList.add("d-none");
                    recordPlaybackContainer.classList.remove("d-none");
                    
                    // Stop all microphone tracks to release the hardware
                    stream.getTracks().forEach(track => track.stop());
                };

                // Start
                mediaRecorder.start();
                recordStartTime = Date.now();
                updateTimer();
                timerInterval = setInterval(updateTimer, 1000);
                startVolumeMonitor(stream);

                // UI Changes
                recordIcon.className = "bi bi-stop-fill";
                recordBtn.classList.add("recording");
                pulseRing.style.opacity = "1";
                recordStatus.innerText = "Recording... Click to stop";
                recordDuration.classList.remove("d-none");
                recordPlaybackContainer.classList.add("d-none");

            } catch (err) {
                console.error("Microphone access denied:", err);
                alert("Microphone Access Required: Please allow microphone permission to record audio.");
                recordStatus.innerText = "Microphone access denied.";
            }
        } else {
            // Stop recording
            mediaRecorder.stop();
            clearInterval(timerInterval);
            stopVolumeMonitor();
            
            // UI Changes
            recordIcon.className = "bi bi-mic";
            recordBtn.classList.remove("recording");
            pulseRing.style.opacity = "0";
        }
    });

    function updateTimer() {
        const diff = Date.now() - recordStartTime;
        const totalSecs = Math.floor(diff / 1000);
        const mins = String(Math.floor(totalSecs / 60)).padStart(2, '0');
        const secs = String(totalSecs % 60).padStart(2, '0');
        recordDuration.innerText = `${mins}:${secs}`;
    }

    function resetRecordState() {
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.stop();
        }
        clearInterval(timerInterval);
        audioChunks = [];
        recordedBlob = null;
        recordIcon.className = "bi bi-mic";
        recordBtn.classList.remove("recording");
        if (pulseRing) pulseRing.style.opacity = "0";
        recordStatus.innerText = "Press the button to start recording";
        recordDuration.classList.add("d-none");
        recordPlaybackContainer.classList.add("d-none");
        audioPlayback.removeAttribute("src");
        audioPlayback.load();
        stopVolumeMonitor();
    }

    function startVolumeMonitor(stream) {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source = audioContext.createMediaStreamSource(stream);
            source.connect(analyser);
            
            const bufferLength = analyser.frequencyBinCount;
            dataArray = new Uint8Array(bufferLength);
            
            let silentFrames = 0;
            let totalFrames = 0;
            
            volumeCheckInterval = setInterval(() => {
                if (!analyser) return;
                analyser.getByteFrequencyData(dataArray);
                let sum = 0;
                for (let i = 0; i < bufferLength; i++) {
                    sum += dataArray[i];
                }
                let averageVolume = sum / bufferLength;
                totalFrames++;
                if (averageVolume < 2.0) { // Silence threshold
                    silentFrames++;
                }
                
                // If recording for >1.5 seconds and >85% is silent, show mic warning
                if (totalFrames > 3 && (silentFrames / totalFrames) > 0.85) {
                    recordStatus.innerHTML = `<span style="color: #ff5c5c; font-weight: bold;"><i class="bi bi-exclamation-triangle-fill me-1"></i> Warning: Extremely low mic input. Check settings.</span>`;
                }
            }, 500);
        } catch (e) {
            console.error("Failed to start volume monitor:", e);
        }
    }

    function stopVolumeMonitor() {
        if (volumeCheckInterval) {
            clearInterval(volumeCheckInterval);
            volumeCheckInterval = null;
        }
        if (source) {
            try { source.disconnect(); } catch(e){}
            source = null;
        }
        if (audioContext) {
            try { audioContext.close(); } catch(e){}
            audioContext = null;
        }
    }

    // ----------------------------------------------------
    // Audio Upload Functionality
    // ----------------------------------------------------
    // Drag and Drop Events
    dropZone.addEventListener("click", () => audioFileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    audioFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    removeFileBtn.addEventListener("click", () => {
        resetUploadState();
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith("audio/")) {
            alert("Invalid File Type: Please upload a valid audio file.");
            return;
        }
        
        // Check size limit: 25MB
        const maxSize = 25 * 1024 * 1024;
        if (file.size > maxSize) {
            alert("File Too Large: Maximum allowed file size is 25MB.");
            return;
        }

        selectedFile = file;
        fileNameText.innerText = file.name;
        fileSizeText.innerText = (file.size / (1024 * 1024)).toFixed(2) + " MB";
        
        const fileURL = URL.createObjectURL(file);
        filePlayback.src = fileURL;
        filePlayback.load();

        dropZone.classList.add("d-none");
        fileUploadDetails.classList.remove("d-none");
    }

    function resetUploadState() {
        selectedFile = null;
        audioFileInput.value = "";
        dropZone.classList.remove("d-none");
        fileUploadDetails.classList.add("d-none");
        filePlayback.removeAttribute("src");
        filePlayback.load();
    }

    // ----------------------------------------------------
    // Result Textarea Utilities
    // ----------------------------------------------------
    function resetResults() {
        transcriptionOutput.value = "";
        statusBadge.classList.add("d-none");
        copyTextBtn.disabled = true;
        downloadFileBtn.setAttribute("disabled", "true");
        downloadFileBtn.removeAttribute("href");
        downloadFileBtn.classList.add("disabled");
    }

    copyTextBtn.addEventListener("click", () => {
        if (!transcriptionOutput.value) return;
        
        navigator.clipboard.writeText(transcriptionOutput.value)
            .then(() => {
                const originalText = copyTextBtn.innerHTML;
                copyTextBtn.innerHTML = `<i class="bi bi-check2"></i> <span>Copied!</span>`;
                setTimeout(() => {
                    copyTextBtn.innerHTML = originalText;
                }, 2000);
            })
            .catch(err => {
                console.error("Could not copy text: ", err);
            });
    });

    // ----------------------------------------------------
    // Submit / API Integration
    // ----------------------------------------------------
    transcribeRecordBtn.addEventListener("click", () => {
        if (!recordedBlob) return;
        submitAudio(recordedBlob, "recorded_speech.webm");
    });

    transcribeUploadBtn.addEventListener("click", () => {
        if (!selectedFile) return;
        submitAudio(selectedFile, selectedFile.name);
    });

    function submitAudio(audioBlobOrFile, filename) {
        // Show Loading State
        resetResults();
        transcriptionLoader.classList.remove("d-none");
        resultContainer.classList.add("d-none");

        // Prepare FormData
        const formData = new FormData();
        formData.append("audio", audioBlobOrFile, filename);
        
        // Append optional api parameters
        if (apiKeyInput.value.trim() !== "") {
            formData.append("api_key", apiKeyInput.value.trim());
        }
        if (modelInput.value.trim() !== "") {
            formData.append("model_name", modelInput.value.trim());
        }

        // Send AJAX Request
        fetch("/transcribe", {
            method: "POST",
            body: formData
        })
        .then(async response => {
            if (response.redirected) {
                // The session has expired or server restarted, redirect browser to login
                window.location.href = response.url;
                return;
            }
            if (!response.ok) {
                let errMsg = `Server responded with status ${response.status}`;
                try {
                    const errData = await response.json();
                    if (errData && errData.error) {
                        errMsg = errData.error;
                    }
                } catch (e) {
                    // JSON parsing failed or response is not JSON
                }
                throw new Error(errMsg);
            }
            return response.json();
        })
        .then(data => {
            // Success response
            transcriptionLoader.classList.add("d-none");
            resultContainer.classList.remove("d-none");
            
            if (data.transcription !== undefined && data.transcription !== null) {
                transcriptionOutput.value = data.transcription || "No Speech detected";
                statusBadge.classList.remove("d-none");
                
                if (data.transcription) {
                    // Enable Actions
                    copyTextBtn.disabled = false;
                    
                    // Setup download link
                    const textBlob = new Blob([data.transcription], { type: "text/plain;charset=utf-8" });
                    const downloadURL = URL.createObjectURL(textBlob);
                    downloadFileBtn.href = downloadURL;
                    downloadFileBtn.setAttribute("download", "tamil_transcription.txt");
                    downloadFileBtn.removeAttribute("disabled");
                    downloadFileBtn.classList.remove("disabled");
                }
            } else {
                transcriptionOutput.value = "An unknown error occurred. No transcription returned.";
            }
        })
        .catch(err => {
            console.error("Transcription failed:", err);
            transcriptionLoader.classList.add("d-none");
            resultContainer.classList.remove("d-none");
            
            // Show errors in transcription text area
            transcriptionOutput.value = `Error Details:\n${err.message}`;
            statusBadge.classList.add("d-none");
        });
    }
});
