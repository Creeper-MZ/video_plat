<!-- app/static/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wan2.1 Video Generation Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            padding-bottom: 50px;
        }
        .navbar {
            background-color: #343a40;
            color: white;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .tab-content {
            padding: 20px;
            background-color: #fff;
            border-radius: 0 0 5px 5px;
            border: 1px solid #dee2e6;
            border-top: none;
        }
        .preview-container {
            max-width: 100%;
            margin-top: 20px;
            text-align: center;
        }
        .preview-container img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .preview-container video {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .task-list {
            margin-top: 20px;
        }
        .task-card {
            margin-bottom: 10px;
        }
        .system-stats {
            font-size: 0.9rem;
        }
        .gpu-card {
            border-left: 5px solid #28a745;
            margin-bottom: 10px;
        }
        .gpu-card.in-use {
            border-left-color: #dc3545;
        }
        .progress {
            height: 25px;
            margin-bottom: 10px;
        }
        .advanced-options {
            margin-top: 20px;
            border-top: 1px solid #eee;
            padding-top: 15px;
        }
        .log-container {
            max-height: 300px;
            overflow-y: auto;
            background-color: #f8f9fa;
            font-family: monospace;
            font-size: 0.85rem;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .log-entry {
            margin: 0;
            padding: 2px 0;
        }
        .log-info {
            color: #0d6efd;
        }
        .log-warning {
            color: #fd7e14;
        }
        .log-error {
            color: #dc3545;
        }
        .log-debug {
            color: #6c757d;
        }
        @media (max-width: 768px) {
            .card {
                margin-bottom: 15px;
            }
            .preview-container img,
            .preview-container video {
                max-width: 100%;
                height: auto;
            }
            .progress {
                height: 20px;
            }
            .log-container {
                max-height: 200px;
            }
        }
        .tooltip-inner {
            max-width: 300px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-4">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="fas fa-film me-2"></i>
                Wan2.1 Video Generation Platform
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="#" id="refreshSystemStatus">
                            <i class="fas fa-sync-alt me-1"></i> Refresh Status
                        </a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container">
        <div class="row">
            <!-- Generation Panel -->
            <div class="col-lg-8">
                <div class="card">
                    <div class="card-header">
                        <ul class="nav nav-tabs card-header-tabs" id="generationTabs" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="t2v-tab" data-bs-toggle="tab" data-bs-target="#t2v" type="button" role="tab">Text to Video</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="i2v-tab" data-bs-toggle="tab" data-bs-target="#i2v" type="button" role="tab">Image to Video</button>
                            </li>
                        </ul>
                    </div>
                    <div class="tab-content" id="generationTabsContent">
                        <!-- Text to Video Tab -->
                        <div class="tab-pane fade show active" id="t2v" role="tabpanel">
                            <form id="t2vForm">
                                <div class="mb-3">
                                    <label for="t2vPrompt" class="form-label">Prompt</label>
                                    <textarea class="form-control" id="t2vPrompt" rows="3" required placeholder="Enter your prompt here..."></textarea>
                                    <div class="form-text">Detailed descriptions generate better results</div>
                                </div>
                                <div class="mb-3">
                                    <label for="t2vNegativePrompt" class="form-label">Negative Prompt</label>
                                    <textarea class="form-control" id="t2vNegativePrompt" rows="2" placeholder="Elements to avoid in generation..."></textarea>
                                    <div class="form-text">Leave empty to use default negative prompt <a href="#" data-bs-toggle="modal" data-bs-target="#defaultNegativePromptModal">View default</a></div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="t2vResolution" class="form-label">Resolution</label>
                                        <select class="form-select" id="t2vResolution">
                                            <option value="832x480" selected>480P Landscape (832x480)</option>
                                            <option value="480x832">480P Portrait (480x832)</option>
                                            <option value="1280x720">720P Landscape (1280x720)</option>
                                            <option value="720x1280">720P Portrait (720x1280)</option>
                                        </select>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="t2vNumFrames" class="form-label">Number of Frames</label>
                                        <input type="number" class="form-control" id="t2vNumFrames" value="81" min="5" step="4">
                                        <div class="form-text">Must be 4n+1 (e.g., 5, 9, 13...)</div>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="t2vFps" class="form-label">FPS</label>
                                        <input type="number" class="form-control" id="t2vFps" value="20" min="1" max="60">
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="t2vSteps" class="form-label">Sampling Steps</label>
                                        <input type="number" class="form-control" id="t2vSteps" value="40" min="20" max="60">
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="t2vSeed" class="form-label">Seed</label>
                                        <input type="number" class="form-control" id="t2vSeed" value="-1">
                                        <div class="form-text">-1 for random seed</div>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="t2vGuideScale" class="form-label">Guidance Scale</label>
                                        <input type="number" class="form-control" id="t2vGuideScale" value="5.0" min="1.0" max="20.0" step="0.1">
                                    </div>
                                </div>

                                <!-- Advanced Options Toggle -->
                                <div class="mb-3">
                                    <a class="btn btn-link p-0" data-bs-toggle="collapse" href="#t2vAdvancedOptions">
                                        <i class="fas fa-cog me-1"></i> Advanced Options
                                    </a>
                                </div>

                                <!-- Advanced Options -->
                                <div class="collapse" id="t2vAdvancedOptions">
                                    <div class="advanced-options">
                                        <div class="row">
                                            <div class="col-md-6 mb-3">
                                                <label for="t2vShift" class="form-label">Sampling Shift</label>
                                                <input type="number" class="form-control" id="t2vShift" value="5.0" min="0.0" max="10.0" step="0.1">
                                            </div>
                                            <div class="col-md-6 mb-3">
                                                <div class="form-check form-switch">
                                                    <input class="form-check-input" type="checkbox" id="t2vUseFp8" checked>
                                                    <label class="form-check-label" for="t2vUseFp8">Use FP8 Quantization</label>
                                                </div>
                                                <div class="form-check form-switch mt-2">
                                                    <input class="form-check-input" type="checkbox" id="t2vSaveVRAM">
                                                    <label class="form-check-label" for="t2vSaveVRAM">Save VRAM</label>
                                                </div>
                                                <div class="form-check form-switch mt-2">
                                                    <input class="form-check-input" type="checkbox" id="t2vDebug">
                                                    <label class="form-check-label" for="t2vDebug">Debug Mode</label>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <button type="submit" class="btn btn-primary mt-3" id="t2vSubmit">
                                    <i class="fas fa-magic me-1"></i> Generate Video
                                </button>
                            </form>
                        </div>

                        <!-- Image to Video Tab -->
                        <div class="tab-pane fade" id="i2v" role="tabpanel">
                            <form id="i2vForm" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <label for="i2vImage" class="form-label">Upload Image</label>
                                    <input class="form-control" type="file" id="i2vImage" accept="image/*" required>
                                </div>
                                <div class="mb-3">
                                    <label for="i2vPrompt" class="form-label">Prompt</label>
                                    <textarea class="form-control" id="i2vPrompt" rows="3" required placeholder="Describe motion and scene details..."></textarea>
                                    <div class="form-text">Good prompts focus on movement and action</div>
                                </div>
                                <div class="mb-3">
                                    <label for="i2vNegativePrompt" class="form-label">Negative Prompt</label>
                                    <textarea class="form-control" id="i2vNegativePrompt" rows="2" placeholder="Elements to avoid in generation..."></textarea>
                                    <div class="form-text">Leave empty to use default negative prompt <a href="#" data-bs-toggle="modal" data-bs-target="#defaultNegativePromptModal">View default</a></div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="i2vResolution" class="form-label">Resolution</label>
                                        <select class="form-select" id="i2vResolution">
                                            <option value="832x480" selected>480P Landscape (832x480)</option>
                                            <option value="480x832">480P Portrait (480x832)</option>
                                            <option value="1280x720">720P Landscape (1280x720)</option>
                                            <option value="720x1280">720P Portrait (720x1280)</option>
                                        </select>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="i2vNumFrames" class="form-label">Number of Frames</label>
                                        <input type="number" class="form-control" id="i2vNumFrames" value="81" min="5" step="4">
                                        <div class="form-text">Must be 4n+1 (e.g., 5, 9, 13...)</div>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="i2vFps" class="form-label">FPS</label>
                                        <input type="number" class="form-control" id="i2vFps" value="20" min="1" max="60">
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="i2vSteps" class="form-label">Sampling Steps</label>
                                        <input type="number" class="form-control" id="i2vSteps" value="40" min="20" max="60">
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="i2vSeed" class="form-label">Seed</label>
                                        <input type="number" class="form-control" id="i2vSeed" value="-1">
                                        <div class="form-text">-1 for random seed</div>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="i2vGuideScale" class="form-label">Guidance Scale</label>
                                        <input type="number" class="form-control" id="i2vGuideScale" value="5.0" min="1.0" max="20.0" step="0.1">
                                    </div>
                                </div>

                                <!-- Advanced Options Toggle -->
                                <div class="mb-3">
                                    <a class="btn btn-link p-0" data-bs-toggle="collapse" href="#i2vAdvancedOptions">
                                        <i class="fas fa-cog me-1"></i> Advanced Options
                                    </a>
                                </div>

                                <!-- Advanced Options -->
                                <div class="collapse" id="i2vAdvancedOptions">
                                    <div class="advanced-options">
                                        <div class="row">
                                            <div class="col-md-6 mb-3">
                                                <label for="i2vShift" class="form-label">Sampling Shift</label>
                                                <input type="number" class="form-control" id="i2vShift" value="5.0" min="0.0" max="10.0" step="0.1">
                                            </div>
                                            <div class="col-md-6 mb-3">
                                                <div class="form-check form-switch">
                                                    <input class="form-check-input" type="checkbox" id="i2vUseFp8" checked>
                                                    <label class="form-check-label" for="i2vUseFp8">Use FP8 Quantization</label>
                                                </div>
                                                <div class="form-check form-switch mt-2">
                                                    <input class="form-check-input" type="checkbox" id="i2vSaveVRAM">
                                                    <label class="form-check-label" for="i2vSaveVRAM">Save VRAM</label>
                                                </div>
                                                <div class="form-check form-switch mt-2">
                                                    <input class="form-check-input" type="checkbox" id="i2vDebug">
                                                    <label class="form-check-label" for="i2vDebug">Debug Mode</label>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div class="mb-3">
                                    <div class="preview-container" id="imagePreviewContainer" style="display: none;">
                                        <p>Image Preview:</p>
                                        <img id="imagePreview" src="#" alt="Preview">
                                    </div>
                                </div>

                                <button type="submit" class="btn btn-primary mt-3" id="i2vSubmit">
                                    <i class="fas fa-magic me-1"></i> Generate Video
                                </button>
                            </form>
                        </div>
                    </div>
                </div>

                <!-- Task Detail Panel -->
                <div class="card" id="taskDetailPanel" style="display: none;">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Task Details</h5>
                        <button type="button" class="btn-close" id="closeTaskDetail"></button>
                    </div>
                    <div class="card-body">
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <p><strong>Task ID:</strong> <span id="detailTaskId"></span></p>
                                <p><strong>Type:</strong> <span id="detailTaskType"></span></p>
                                <p><strong>Status:</strong> <span id="detailTaskStatus" class="badge bg-secondary"></span></p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>Created:</strong> <span id="detailTaskCreated"></span></p>
                                <p><strong>Started:</strong> <span id="detailTaskStarted"></span></p>
                                <p><strong>Completed:</strong> <span id="detailTaskCompleted"></span></p>
                            </div>
                        </div>

                        <div class="progress" id="taskProgressBar">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%">0%</div>
                        </div>

                        <div class="mt-3">
                            <h6>Task Parameters:</h6>
                            <pre id="detailTaskParams" class="bg-light p-2 rounded"></pre>
                        </div>

                        <div class="mt-3">
                            <h6>Log Messages:</h6>
                            <div class="log-container" id="detailTaskLogs"></div>
                        </div>

                        <div class="preview-container mt-3" id="resultPreviewContainer" style="display: none;">
                            <h6>Result:</h6>
                            <video id="videoResult" controls loop style="max-width: 100%"></video>
                        </div>

                        <div class="mt-3">
                            <button class="btn btn-danger" id="cancelTaskBtn">
                                <i class="fas fa-stop-circle me-1"></i> Cancel Task
                            </button>
                            <a class="btn btn-primary" id="downloadResultBtn" href="#" download style="display: none;">
                                <i class="fas fa-download me-1"></i> Download Result
                            </a>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Status Panel -->
            <div class="col-lg-4">
                <!-- System Status -->
                <div class="card mb-3">
                    <div class="card-header">
                        <h5 class="mb-0">System Status</h5>
                    </div>
                    <div class="card-body">
                        <h6>GPU Resources</h6>
                        <div id="gpuStatus">
                            <div class="text-center">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                                <p class="mt-2">Loading GPU status...</p>
                            </div>
                        </div>
                        
                        <h6 class="mt-4">Task Queue</h6>
                        <div id="queueStatus">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p>Loading queue status...</p>
                        </div>
                    </div>
                </div>

                <!-- Tasks List -->
                <div class="card">
                    <div class="card-header">
                        <ul class="nav nav-tabs card-header-tabs" id="tasksTabs" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="queue-tab" data-bs-toggle="tab" data-bs-target="#queuedTasks" type="button" role="tab">Queued</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="running-tab" data-bs-toggle="tab" data-bs-target="#runningTasks" type="button" role="tab">Running</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="completed-tab" data-bs-toggle="tab" data-bs-target="#completedTasks" type="button" role="tab">Completed</button>
                            </li>
                        </ul>
                    </div>
                    <div class="tab-content" id="tasksTabsContent">
                        <div class="tab-pane fade show active" id="queuedTasks" role="tabpanel">
                            <div class="list-group list-group-flush" id="queuedTasksList">
                                <div class="list-group-item text-center py-3">No queued tasks</div>
                            </div>
                        </div>
                        <div class="tab-pane fade" id="runningTasks" role="tabpanel">
                            <div class="list-group list-group-flush" id="runningTasksList">
                                <div class="list-group-item text-center py-3">No running tasks</div>
                            </div>
                        </div>
                        <div class="tab-pane fade" id="completedTasks" role="tabpanel">
                            <div class="list-group list-group-flush" id="completedTasksList">
                                <div class="list-group-item text-center py-3">No completed tasks</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Toast Notifications -->
    <div class="toast-container position-fixed bottom-0 end-0 p-3">
        <div id="toast" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header">
                <strong class="me-auto" id="toastTitle">Notification</strong>
                <small id="toastTime">just now</small>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body" id="toastMessage">
                Message here
            </div>
        </div>
    </div>

    <!-- Default Negative Prompt Modal -->
    <div class="modal fade" id="defaultNegativePromptModal" tabindex="-1" aria-labelledby="defaultNegativePromptModalLabel" aria-hidden="true">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="defaultNegativePromptModalLabel">Default Negative Prompt</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p>The following negative prompt will be used if you don't provide one:</p>
                    <pre class="bg-light p-3 small" style="max-height: 200px; overflow-y: auto;">色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走</pre>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-primary" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Loading Modal -->
    <div class="modal fade" id="loadingModal" tabindex="-1" data-bs-backdrop="static" data-bs-keyboard="false">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-body text-center p-4">
                    <div class="spinner-border text-primary mb-3" role="status" style="width: 3rem; height: 3rem;">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h5 id="loadingModalText">Loading system status...</h5>
                    <p class="text-muted" id="loadingModalSubtext">Please wait while we connect to the server.</p>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Global variables
        let currentTaskId = null;
        let taskRefreshInterval = null;
        let systemRefreshInterval = null;
        const SYSTEM_REFRESH_INTERVAL = 5000; // 5 seconds
        const TASK_REFRESH_INTERVAL = 2000;   // 2 seconds
        
        // Initialize tooltips
        const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
        
        // Image preview
        document.getElementById('i2vImage').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('imagePreview').src = e.target.result;
                    document.getElementById('imagePreviewContainer').style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });
        
        // Form submission handlers
        document.getElementById('t2vForm').addEventListener('submit', function(e) {
            e.preventDefault();
            submitT2VForm();
        });
        
        document.getElementById('i2vForm').addEventListener('submit', function(e) {
            e.preventDefault();
            submitI2VForm();
        });
        
        // Refresh system status button
        document.getElementById('refreshSystemStatus').addEventListener('click', function(e) {
            e.preventDefault();
            fetchSystemStatus();
        });
        
        // Close task detail button
        document.getElementById('closeTaskDetail').addEventListener('click', function() {
            document.getElementById('taskDetailPanel').style.display = 'none';
            if (taskRefreshInterval) {
                clearInterval(taskRefreshInterval);
                taskRefreshInterval = null;
            }
            currentTaskId = null;
        });
        
        // Cancel task button
        document.getElementById('cancelTaskBtn').addEventListener('click', function() {
            if (currentTaskId) {
                cancelTask(currentTaskId);
            }
        });
        
        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            // Show loading modal
            const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
            loadingModal.show();
            
            // Fetch initial system status
            fetchSystemStatus()
                .then(() => {
                    // Hide loading modal after data is loaded
                    loadingModal.hide();
                })
                .catch(error => {
                    document.getElementById('loadingModalText').textContent = 'Connection Error';
                    document.getElementById('loadingModalSubtext').textContent = 
                        'Could not connect to the server. Please check your connection and refresh the page.';
                    
                    // Add a retry button
                    const modalBody = document.querySelector('#loadingModal .modal-body');
                    const retryBtn = document.createElement('button');
                    retryBtn.className = 'btn btn-primary mt-3';
                    retryBtn.textContent = 'Retry Connection';
                    retryBtn.onclick = function() {
                        location.reload();
                    };
                    modalBody.appendChild(retryBtn);
                });
            
            // Set up periodic refresh for system status
            systemRefreshInterval = setInterval(fetchSystemStatus, SYSTEM_REFRESH_INTERVAL);
        });
        
        // Helper functions
        function showToast(title, message, type = 'success') {
            const toast = document.getElementById('toast');
            const toastInstance = new bootstrap.Toast(toast);
            
            document.getElementById('toastTitle').textContent = title;
            document.getElementById('toastMessage').textContent = message;
            document.getElementById('toastTime').textContent = new Date().toLocaleTimeString();
            
            // Set toast color based on type
            toast.classList.remove('bg-success', 'bg-danger', 'bg-warning', 'text-white');
            if (type === 'success') {
                toast.classList.add('bg-success', 'text-white');
            } else if (type === 'error') {
                toast.classList.add('bg-danger', 'text-white');
            } else if (type === 'warning') {
                toast.classList.add('bg-warning');
            }
            
            toastInstance.show();
        }
        
        function formatTimestamp(timestamp) {
            if (!timestamp) return 'N/A';
            return new Date(timestamp).toLocaleString();
        }
        
        function submitT2VForm() {
            const formData = {
                basic: {
                    prompt: document.getElementById('t2vPrompt').value,
                    negative_prompt: document.getElementById('t2vNegativePrompt').value || undefined,
                    resolution: document.getElementById('t2vResolution').value,
                    num_frames: parseInt(document.getElementById('t2vNumFrames').value),
                    fps: parseInt(document.getElementById('t2vFps').value),
                    steps: parseInt(document.getElementById('t2vSteps').value),
                    shift: parseFloat(document.getElementById('t2vShift').value) || undefined,
                    guide_scale: parseFloat(document.getElementById('t2vGuideScale').value),
                    seed: parseInt(document.getElementById('t2vSeed').value),
                    use_fp8: document.getElementById('t2vUseFp8').checked
                },
                advanced: {
                    save_vram: document.getElementById('t2vSaveVRAM').checked,
                    debug: document.getElementById('t2vDebug').checked
                }
            };
            
            // Disable submit button
            const submitBtn = document.getElementById('t2vSubmit');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Submitting...';
            submitBtn.disabled = true;
            
            fetch('/api/generate/t2v', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.detail || 'Server error');
                    });
                }
                return response.json();
            })
            .then(data => {
                showToast('Success', `Task ${data.task_id} added to queue`);
                // Open task detail panel for the new task
                openTaskDetail(data.task_id);
                
                // Force refresh system status
                fetchSystemStatus();
            })
            .catch(error => {
                showToast('Error', `Failed to submit task: ${error.message}`, 'error');
            })
            .finally(() => {
                // Re-enable submit button
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
            });
        }
        
        function submitI2VForm() {
            const formData = new FormData();
            
            // Get file
            const imageFile = document.getElementById('i2vImage').files[0];
            if (!imageFile) {
                showToast('Error', 'Please select an image file', 'error');
                return;
            }
            
            // Add form fields
            formData.append('prompt', document.getElementById('i2vPrompt').value);
            formData.append('negative_prompt', document.getElementById('i2vNegativePrompt').value || '');
            formData.append('resolution', document.getElementById('i2vResolution').value);
            formData.append('num_frames', document.getElementById('i2vNumFrames').value);
            formData.append('fps', document.getElementById('i2vFps').value);
            formData.append('steps', document.getElementById('i2vSteps').value);
            formData.append('shift', document.getElementById('i2vShift').value || '');
            formData.append('guide_scale', document.getElementById('i2vGuideScale').value);
            formData.append('seed', document.getElementById('i2vSeed').value);
            formData.append('use_fp8', document.getElementById('i2vUseFp8').checked);
            formData.append('save_vram', document.getElementById('i2vSaveVRAM').checked);
            formData.append('debug', document.getElementById('i2vDebug').checked);
            formData.append('image', imageFile);
            
            // Disable submit button
            const submitBtn = document.getElementById('i2vSubmit');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Submitting...';
            submitBtn.disabled = true;
            
            fetch('/api/generate/i2v', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.detail || 'Server error');
                    });
                }
                return response.json();
            })
            .then(data => {
                showToast('Success', `Task ${data.task_id} added to queue`);
                // Open task detail panel for the new task
                openTaskDetail(data.task_id);
                
                // Force refresh system status
                fetchSystemStatus();
            })
            .catch(error => {
                showToast('Error', `Failed to submit task: ${error.message}`, 'error');
            })
            .finally(() => {
                // Re-enable submit button
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
            });
        }
        
        function fetchSystemStatus() {
            // Fetch GPU and queue status
            return fetch('/api/status/system')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to fetch system status');
                    }
                    return response.json();
                })
                .then(data => {
                    updateGPUStatus(data.gpus);
                    updateQueueStatus(data.queue);
                    updateTaskLists(data.queue);
                    return data; // Return data for promise chaining
                })
                .catch(error => {
                    console.error('Error fetching system status:', error);
                    showToast('Error', 'Failed to update system status. Retrying...', 'error');
                    throw error; // Re-throw for promise chaining
                });
        }
        
        function updateGPUStatus(gpus) {
            const gpuStatusEl = document.getElementById('gpuStatus');
            if (!gpus || gpus.length === 0) {
                gpuStatusEl.innerHTML = '<p class="text-center">No GPU information available</p>';
                return;
            }
            
            let html = '';
            gpus.forEach(gpu => {
                const isInUse = !gpu.available;
                const usedClass = isInUse ? 'in-use' : '';
                const usagePercent = gpu.vram_total ? Math.floor((gpu.vram_used / gpu.vram_total) * 100) : 0;
                
                html += `
                <div class="gpu-card card p-2 mb-2 ${usedClass}">
                    <div class="d-flex justify-content-between">
                        <div>
                            <strong>GPU ${gpu.device_id}</strong>
                            ${isInUse ? `<span class="badge bg-danger ms-2">In Use</span>` : '<span class="badge bg-success ms-2">Available</span>'}
                        </div>
                        <div class="text-end">
                            <small>${isInUse ? `Task: ${gpu.current_task}` : 'Idle'}</small>
                        </div>
                    </div>
                    ${gpu.vram_used !== undefined ? `
                    <div class="mt-2">
                        <div class="d-flex justify-content-between small text-muted">
                            <span>VRAM Usage: ${gpu.vram_used}MB / ${gpu.vram_total}MB</span>
                            <span>${usagePercent}%</span>
                        </div>
                        <div class="progress mt-1" style="height: 5px;">
                            <div class="progress-bar" role="progressbar" style="width: ${usagePercent}%"></div>
                        </div>
                    </div>
                    ` : ''}
                    ${gpu.utilization !== undefined ? `
                    <div class="small text-muted mt-1">
                        Utilization: ${gpu.utilization.toFixed(1)}%
                    </div>
                    ` : ''}
                </div>
                `;
            });
            
            gpuStatusEl.innerHTML = html;
        }
        
        function updateQueueStatus(queueData) {
            const queueStatusEl = document.getElementById('queueStatus');
            if (!queueData) {
                queueStatusEl.innerHTML = '<p class="text-center">No queue information available</p>';
                return;
            }
            
            const html = `
            <ul class="list-group">
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    Queued Tasks
                    <span class="badge bg-primary rounded-pill">${queueData.queue_length}</span>
                </li>
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    Running Tasks
                    <span class="badge bg-success rounded-pill">${queueData.running_tasks}</span>
                </li>
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    Completed Tasks
                    <span class="badge bg-secondary rounded-pill">${queueData.completed_tasks}</span>
                </li>
            </ul>
            `;
            
            queueStatusEl.innerHTML = html;
        }
        
        function updateTaskLists(queueData) {
            // Update queued tasks
            const queuedListEl = document.getElementById('queuedTasksList');
            if (queueData.queue.length === 0) {
                queuedListEl.innerHTML = '<div class="list-group-item text-center py-3">No queued tasks</div>';
            } else {
                let html = '';
                queueData.queue.forEach(task => {
                    html += getTaskListItemHtml(task);
                });
                queuedListEl.innerHTML = html;
            }
            
            // Update running tasks
            const runningListEl = document.getElementById('runningTasksList');
            if (queueData.running.length === 0) {
                runningListEl.innerHTML = '<div class="list-group-item text-center py-3">No running tasks</div>';
            } else {
                let html = '';
                queueData.running.forEach(task => {
                    html += getTaskListItemHtml(task);
                });
                runningListEl.innerHTML = html;
            }
            
            // Update completed tasks
            const completedListEl = document.getElementById('completedTasksList');
            if (queueData.recent_completed.length === 0) {
                completedListEl.innerHTML = '<div class="list-group-item text-center py-3">No completed tasks</div>';
            } else {
                let html = '';
                queueData.recent_completed.forEach(task => {
                    html += getTaskListItemHtml(task);
                });
                completedListEl.innerHTML = html;
            }
            
            // Add event listeners to task items
            document.querySelectorAll('.task-list-item').forEach(item => {
                item.addEventListener('click', function() {
                    const taskId = this.getAttribute('data-task-id');
                    openTaskDetail(taskId);
                });
            });
        }
        
        function getTaskListItemHtml(task) {
            let statusBadge = '';
            switch (task.status) {
                case 'queued':
                    statusBadge = '<span class="badge bg-secondary">Queued</span>';
                    break;
                case 'running':
                    statusBadge = '<span class="badge bg-primary">Running</span>';
                    break;
                case 'completed':
                    statusBadge = '<span class="badge bg-success">Completed</span>';
                    break;
                case 'failed':
                    statusBadge = '<span class="badge bg-danger">Failed</span>';
                    break;
                case 'cancelled':
                    statusBadge = '<span class="badge bg-warning text-dark">Cancelled</span>';
                    break;
                default:
                    statusBadge = `<span class="badge bg-secondary">${task.status}</span>`;
            }
            
            return `
            <a href="#" class="list-group-item list-group-item-action task-list-item" data-task-id="${task.id}">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">Task ${task.id.substring(0, 8)}...</h6>
                    ${statusBadge}
                </div>
                <p class="mb-1 small text-truncate">${task.params.prompt || 'No prompt'}</p>
                <div class="d-flex w-100 justify-content-between align-items-center">
                    <small class="text-muted">${formatTimestamp(task.created_at)}</small>
                    ${task.status === 'running' ? `<div class="small">${task.progress}%</div>` : ''}
                </div>
                ${task.status === 'running' ? `
                <div class="progress mt-1" style="height: 5px;">
                    <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: ${task.progress}%"></div>
                </div>
                ` : ''}
            </a>
            `;
        }
        
        function openTaskDetail(taskId) {
            // Set current task ID
            currentTaskId = taskId;
            
            // Show the task detail panel
            document.getElementById('taskDetailPanel').style.display = 'block';
            
            // Fetch and display initial task info
            fetchTaskDetail(taskId);
            
            // Set up periodic refresh
            if (taskRefreshInterval) {
                clearInterval(taskRefreshInterval);
            }
            taskRefreshInterval = setInterval(() => fetchTaskDetail(taskId), TASK_REFRESH_INTERVAL);
        }
        
        function fetchTaskDetail(taskId) {
            fetch(`/api/generate/task/${taskId}`)
                .then(response => {
                    if (!response.ok) {
                        if (response.status === 404) {
                            throw new Error('Task not found');
                        }
                        return response.json().then(data => {
                            throw new Error(data.detail || 'Failed to fetch task details');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    updateTaskDetail(data);
                })
                .catch(error => {
                    console.error(`Error fetching task ${taskId}:`, error);
                    
                    if (taskRefreshInterval && 
                        (error.message.includes('not found'))) {
                        // Task may have been deleted, stop refreshing
                        clearInterval(taskRefreshInterval);
                        taskRefreshInterval = null;
                        showToast('Error', `Task ${taskId} not found`, 'error');
                    }
                });
        }
        
        function updateTaskDetail(task) {
            // Basic info
            document.getElementById('detailTaskId').textContent = task.id;
            document.getElementById('detailTaskType').textContent = task.type.toUpperCase();
            
            // Status
            const statusEl = document.getElementById('detailTaskStatus');
            statusEl.textContent = task.status.charAt(0).toUpperCase() + task.status.slice(1);
            statusEl.className = 'badge';  // Reset class
            
            switch (task.status) {
                case 'queued':
                    statusEl.classList.add('bg-secondary');
                    break;
                case 'running':
                    statusEl.classList.add('bg-primary');
                    break;
                case 'completed':
                    statusEl.classList.add('bg-success');
                    break;
                case 'failed':
                    statusEl.classList.add('bg-danger');
                    break;
                case 'cancelled':
                    statusEl.classList.add('bg-warning', 'text-dark');
                    break;
                default:
                    statusEl.classList.add('bg-secondary');
            }
            
            // Timestamps
            document.getElementById('detailTaskCreated').textContent = formatTimestamp(task.created_at);
            document.getElementById('detailTaskStarted').textContent = formatTimestamp(task.started_at);
            document.getElementById('detailTaskCompleted').textContent = formatTimestamp(task.completed_at);
            
            // Progress bar
            const progressBar = document.getElementById('taskProgressBar').querySelector('.progress-bar');
            progressBar.style.width = `${task.progress}%`;
            progressBar.textContent = `${task.progress}%`;
            
            // Parameters
            document.getElementById('detailTaskParams').textContent = JSON.stringify(task.params, null, 2);
            
            // Logs
            const logsContainer = document.getElementById('detailTaskLogs');
            if (task.logs && task.logs.length > 0) {
                let logsHtml = '';
                task.logs.forEach(log => {
                    const logClass = `log-${log.level}`;
                    const timestamp = new Date(log.time).toLocaleTimeString();
                    logsHtml += `<p class="log-entry ${logClass}">[${timestamp}] ${log.message}</p>`;
                });
                logsContainer.innerHTML = logsHtml;
                
                // Scroll to bottom of logs if not scrolled up
                if (logsContainer.scrollTop + logsContainer.clientHeight >= logsContainer.scrollHeight - 50) {
                    logsContainer.scrollTop = logsContainer.scrollHeight;
                }
            } else {
                logsContainer.innerHTML = '<p class="text-center text-muted">No logs available</p>';
            }
            
            // Result preview
            const resultPreviewContainer = document.getElementById('resultPreviewContainer');
            const downloadBtn = document.getElementById('downloadResultBtn');
            
            if (task.status === 'completed' && task.result && task.result.file_url) {
                resultPreviewContainer.style.display = 'block';
                const videoResult = document.getElementById('videoResult');
                videoResult.src = task.result.file_url;
                
                // Download button
                downloadBtn.style.display = 'inline-block';
                downloadBtn.href = task.result.file_url;
                downloadBtn.download = task.result.file_url.split('/').pop();
            } else {
                resultPreviewContainer.style.display = 'none';
                downloadBtn.style.display = 'none';
            }
            
            // Cancel button visibility
            const cancelBtn = document.getElementById('cancelTaskBtn');
            if (task.status === 'queued' || task.status === 'running') {
                cancelBtn.style.display = 'inline-block';
            } else {
                cancelBtn.style.display = 'none';
            }
            
            // Stop refreshing if task is complete/failed/cancelled
            if (['completed', 'failed', 'cancelled'].includes(task.status) && taskRefreshInterval) {
                clearInterval(taskRefreshInterval);
                taskRefreshInterval = null;
            }
        }
        
        function cancelTask(taskId) {
            if (!confirm(`Are you sure you want to cancel task ${taskId}?`)) {
                return;
            }
            
            fetch(`/api/generate/task/${taskId}`, {
                method: 'DELETE'
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.detail || 'Failed to cancel task');
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'cancelled') {
                    showToast('Success', `Task ${taskId} cancelled successfully`);
                    // Refresh task details
                    fetchTaskDetail(taskId);
                    // Refresh system status
                    fetchSystemStatus();
                } else {
                    showToast('Warning', data.message || 'Task status is not cancelled', 'warning');
                }
            })
            .catch(error => {
                showToast('Error', `Failed to cancel task: ${error.message}`, 'error');
            });
        }
    </script>
</body>
</html>