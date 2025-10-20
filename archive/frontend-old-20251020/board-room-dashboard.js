/**
 * Board Room Dashboard JavaScript
 * Real-time Echo Brain Board of Directors interface
 */

class BoardRoomDashboard {
    constructor() {
        this.websocket = null;
        this.currentTaskId = null;
        this.currentTaskStatus = null;
        this.directors = {};
        this.timeline = [];
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;

        // Configuration
        this.config = {
            wsUrl: `ws://${window.location.hostname}:8309/api/board/ws`,
            apiUrl: `http://${window.location.hostname}:8309/api/board`,
            reconnectDelay: 3000
        };

        // Initialize dashboard
        this.init();
    }

    async init() {
        console.log('Initializing Board Room Dashboard...');

        // Load initial data
        await this.loadInitialData();

        // Connect WebSocket
        this.connectWebSocket();

        // Set up event listeners
        this.setupEventListeners();

        // Start periodic updates
        this.startPeriodicUpdates();

        this.addLog('System initialized successfully', 'success');
    }

    async loadInitialData() {
        try {
            // Load board status
            const statusResponse = await fetch(`${this.config.apiUrl}/status`);
            if (statusResponse.ok) {
                const statusData = await statusResponse.json();
                this.updateBoardStatus(statusData);
            }

            // Load directors list
            const directorsResponse = await fetch(`${this.config.apiUrl}/directors`);
            if (directorsResponse.ok) {
                const directorsData = await directorsResponse.json();
                this.updateDirectorsList(directorsData.directors);
            }

        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.addLog('Failed to load initial data: ' + error.message, 'error');
        }
    }

    connectWebSocket() {
        try {
            this.websocket = new WebSocket(this.config.wsUrl);

            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected');
                this.addLog('Real-time connection established', 'success');
            };

            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus('disconnected');
                this.addLog('Real-time connection lost', 'warning');
                this.scheduleReconnect();
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.addLog('Connection error: ' + error.message, 'error');
            };

        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.addLog('Failed to establish connection: ' + error.message, 'error');
            this.scheduleReconnect();
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                console.log(`Reconnection attempt ${this.reconnectAttempts}`);
                this.connectWebSocket();
            }, this.config.reconnectDelay);
        } else {
            this.addLog('Max reconnection attempts reached', 'error');
            this.updateConnectionStatus('failed');
        }
    }

    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('WebSocket message received:', data);

            switch (data.type) {
                case 'task_submitted':
                    this.handleTaskSubmitted(data);
                    break;
                case 'director_started':
                    this.handleDirectorStarted(data);
                    break;
                case 'director_completed':
                    this.handleDirectorCompleted(data);
                    break;
                case 'director_error':
                    this.handleDirectorError(data);
                    break;
                case 'task_completed':
                    this.handleTaskCompleted(data);
                    break;
                case 'task_error':
                    this.handleTaskError(data);
                    break;
                case 'user_override':
                    this.handleUserOverride(data);
                    break;
                case 'pong':
                    // Handle ping/pong for connection keepalive
                    break;
                default:
                    console.log('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error handling WebSocket message:', error);
            this.addLog('Error processing real-time update', 'error');
        }
    }

    handleTaskSubmitted(data) {
        this.currentTaskId = data.task_id;
        this.currentTaskStatus = 'pending';

        this.addTimelineItem({
            timestamp: data.timestamp,
            status: 'pending',
            description: `Task submitted: ${data.description.substring(0, 100)}...`,
            details: `Task ID: ${data.task_id}`
        });

        this.addActivity('task', 'New Task Submitted',
            `Task ${data.task_id.substring(0, 8)} submitted by ${data.user_id}`,
            data.timestamp);

        this.updateTaskControls(true);
        this.addLog(`Task ${data.task_id.substring(0, 8)} submitted for evaluation`, 'info');
        this.showNotification('Task submitted successfully', 'success');
    }

    handleDirectorStarted(data) {
        const directorId = data.director_id;
        if (this.directors[directorId]) {
            this.directors[directorId].status = 'evaluating';
            this.directors[directorId].current_task = data.task_id;
            this.updateDirectorCard(directorId, this.directors[directorId]);
        }

        this.addTimelineItem({
            timestamp: data.timestamp,
            status: 'in_progress',
            description: `${data.director_name} started evaluation`,
            details: `Director ${data.director_name} is now evaluating the task`
        });

        this.addActivity('director', 'Director Evaluation Started',
            `${data.director_name} began task evaluation`,
            data.timestamp);

        this.addLog(`${data.director_name} started evaluation`, 'info');
    }

    handleDirectorCompleted(data) {
        const directorId = data.director_id;
        if (this.directors[directorId]) {
            this.directors[directorId].status = 'completed';
            this.directors[directorId].last_recommendation = data.recommendation;
            this.directors[directorId].last_confidence = data.confidence;
            this.updateDirectorCard(directorId, this.directors[directorId]);
        }

        this.addTimelineItem({
            timestamp: data.timestamp,
            status: 'completed',
            description: `${data.director_name} completed evaluation`,
            details: `Recommendation: ${data.recommendation} (${(data.confidence * 100).toFixed(1)}% confidence)`
        });

        this.addActivity('director', 'Director Evaluation Completed',
            `${data.director_name}: ${data.recommendation} (${(data.confidence * 100).toFixed(1)}%)`,
            data.timestamp);

        this.addLog(`${data.director_name} completed: ${data.recommendation}`, 'success');

        // Update consensus if we have task details
        this.updateConsensusDisplay();
    }

    handleDirectorError(data) {
        const directorId = data.director_id;
        if (this.directors[directorId]) {
            this.directors[directorId].status = 'error';
            this.directors[directorId].last_error = data.error;
            this.updateDirectorCard(directorId, this.directors[directorId]);
        }

        this.addTimelineItem({
            timestamp: data.timestamp,
            status: 'error',
            description: `${data.director_id} evaluation failed`,
            details: `Error: ${data.error}`
        });

        this.addActivity('system', 'Director Error',
            `${data.director_id} evaluation failed: ${data.error}`,
            data.timestamp);

        this.addLog(`${data.director_id} failed: ${data.error}`, 'error');
    }

    handleTaskCompleted(data) {
        this.currentTaskStatus = 'completed';

        this.addTimelineItem({
            timestamp: data.timestamp,
            status: 'completed',
            description: 'Task evaluation completed',
            details: `Final recommendation: ${data.final_recommendation}`
        });

        this.addActivity('task', 'Task Completed',
            `Evaluation complete: ${data.final_recommendation}`,
            data.timestamp);

        this.updateTaskControls(false);
        this.addLog(`Task completed: ${data.final_recommendation}`, 'success');
        this.showNotification('Task evaluation completed!', 'success');

        // Update consensus with final results
        this.updateConsensusDisplay();
    }

    handleTaskError(data) {
        this.currentTaskStatus = 'error';

        this.addTimelineItem({
            timestamp: data.timestamp,
            status: 'error',
            description: 'Task evaluation failed',
            details: `Error: ${data.error}`
        });

        this.addActivity('system', 'Task Error',
            `Task ${data.task_id.substring(0, 8)} failed: ${data.error}`,
            data.timestamp);

        this.updateTaskControls(false);
        this.addLog(`Task failed: ${data.error}`, 'error');
        this.showNotification('Task evaluation failed', 'error');
    }

    handleUserOverride(data) {
        this.addTimelineItem({
            timestamp: data.timestamp,
            status: 'overridden',
            description: `User override: ${data.override_type}`,
            details: `User provided override with type: ${data.override_type}`
        });

        this.addActivity('user', 'User Override',
            `User ${data.override_type} the recommendation`,
            data.timestamp);

        this.addLog(`User override: ${data.override_type}`, 'warning');
        this.showNotification('Feedback submitted successfully', 'success');
    }

    async updateConsensusDisplay() {
        if (!this.currentTaskId) return;

        try {
            const response = await fetch(`${this.config.apiUrl}/decisions/${this.currentTaskId}`);
            if (response.ok) {
                const taskData = await response.json();

                const consensusPercent = Math.round(taskData.consensus_score * 100);
                const consensusDegrees = (taskData.consensus_score * 360);

                document.getElementById('consensus-value').textContent = `${consensusPercent}%`;
                document.getElementById('consensus-circle').style.setProperty('--progress', `${consensusDegrees}deg`);

                document.getElementById('confidence-score').textContent = taskData.confidence_score.toFixed(2);
                document.getElementById('director-count').textContent = taskData.director_count;
                document.getElementById('evidence-count').textContent = taskData.evidence_count;
            }
        } catch (error) {
            console.error('Failed to update consensus:', error);
        }
    }

    updateBoardStatus(statusData) {
        document.getElementById('task-count-text').textContent =
            `${statusData.active_tasks} Active Tasks`;

        // Update director status
        for (const [directorId, directorData] of Object.entries(statusData.director_status)) {
            this.directors[directorId] = {
                ...this.directors[directorId],
                ...directorData,
                status: directorData.status || 'idle'
            };
        }
    }

    updateDirectorsList(directors) {
        const container = document.getElementById('directors-container');
        container.innerHTML = '';

        directors.forEach(director => {
            this.directors[director.director_id] = director;
            const directorCard = this.createDirectorCard(director);
            container.appendChild(directorCard);
        });
    }

    createDirectorCard(director) {
        const card = document.createElement('div');
        card.className = `director-card ${director.status || 'idle'}`;
        card.setAttribute('data-director-id', director.director_id);
        card.onclick = () => this.showDirectorDetails(director.director_id);

        card.innerHTML = `
            <div class="director-header">
                <div class="director-name">${director.director_name}</div>
                <div class="director-status ${director.status || 'idle'}">${director.status || 'idle'}</div>
            </div>
            <div class="director-details">${director.specialization}</div>
            <div class="director-metrics">
                <span>Load: ${director.current_load || 0}</span>
                <span>Avg Time: ${(director.average_response_time || 0).toFixed(1)}s</span>
                <span>Approval: ${((director.approval_rate || 0) * 100).toFixed(0)}%</span>
            </div>
        `;

        return card;
    }

    updateDirectorCard(directorId, directorData) {
        const card = document.querySelector(`[data-director-id="${directorId}"]`);
        if (card) {
            // Update status class
            card.className = `director-card ${directorData.status || 'idle'}`;

            // Update status text
            const statusElement = card.querySelector('.director-status');
            if (statusElement) {
                statusElement.textContent = directorData.status || 'idle';
                statusElement.className = `director-status ${directorData.status || 'idle'}`;
            }

            // Update metrics if available
            if (directorData.last_confidence !== undefined) {
                const metricsElement = card.querySelector('.director-metrics');
                if (metricsElement) {
                    metricsElement.innerHTML = `
                        <span>Load: ${directorData.current_load || 0}</span>
                        <span>Confidence: ${(directorData.last_confidence * 100).toFixed(0)}%</span>
                        <span>Recommendation: ${directorData.last_recommendation || 'N/A'}</span>
                    `;
                }
            }
        }
    }

    addTimelineItem(item) {
        this.timeline.unshift(item); // Add to beginning
        if (this.timeline.length > 20) {
            this.timeline.pop(); // Keep only last 20 items
        }

        const container = document.getElementById('timeline-container');
        const timelineItem = document.createElement('div');
        timelineItem.className = `timeline-item ${item.status}`;

        const timeStr = new Date(item.timestamp).toLocaleTimeString();
        timelineItem.innerHTML = `
            <div class="timeline-time">${timeStr}</div>
            <div class="timeline-content">${item.description}</div>
            <div class="timeline-details">${item.details || ''}</div>
        `;

        container.insertBefore(timelineItem, container.firstChild);

        // Remove excess items from DOM
        while (container.children.length > 20) {
            container.removeChild(container.lastChild);
        }
    }

    addActivity(type, title, description, timestamp) {
        const container = document.getElementById('activity-container');
        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';

        const timeStr = new Date(timestamp).toLocaleTimeString();
        activityItem.innerHTML = `
            <div class="activity-icon ${type}">
                <i class="fas fa-${this.getActivityIcon(type)}"></i>
            </div>
            <div class="activity-content">
                <div class="activity-title">${title}</div>
                <div class="activity-description">${description}</div>
            </div>
            <div class="activity-time">${timeStr}</div>
        `;

        container.insertBefore(activityItem, container.firstChild);

        // Keep only last 10 activities
        while (container.children.length > 10) {
            container.removeChild(container.lastChild);
        }
    }

    getActivityIcon(type) {
        const icons = {
            task: 'tasks',
            director: 'user-tie',
            user: 'user',
            system: 'cog'
        };
        return icons[type] || 'info-circle';
    }

    addLog(message, type = 'info') {
        const container = document.getElementById('logs-container');
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;

        const timestamp = new Date().toLocaleTimeString();
        logEntry.textContent = `[${timestamp}] ${message}`;

        container.appendChild(logEntry);
        container.scrollTop = container.scrollHeight;

        // Keep only last 50 log entries
        while (container.children.length > 50) {
            container.removeChild(container.firstChild);
        }
    }

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        const textElement = document.getElementById('connection-text');

        switch (status) {
            case 'connected':
                statusElement.className = 'status-indicator active';
                textElement.textContent = 'Connected';
                break;
            case 'disconnected':
                statusElement.className = 'status-indicator warning';
                textElement.textContent = 'Disconnected';
                break;
            case 'failed':
                statusElement.className = 'status-indicator error';
                textElement.textContent = 'Connection Failed';
                break;
            default:
                statusElement.className = 'status-indicator';
                textElement.textContent = 'Connecting...';
        }
    }

    updateTaskControls(hasActiveTask) {
        const modifyBtn = document.getElementById('modify-btn');
        const approveBtn = document.getElementById('approve-btn');
        const rejectBtn = document.getElementById('reject-btn');

        if (hasActiveTask && this.currentTaskStatus === 'completed') {
            modifyBtn.disabled = false;
            approveBtn.disabled = false;
            rejectBtn.disabled = false;
        } else {
            modifyBtn.disabled = true;
            approveBtn.disabled = true;
            rejectBtn.disabled = true;
        }
    }

    async showDirectorDetails(directorId) {
        const director = this.directors[directorId];
        if (!director) return;

        const modal = document.getElementById('director-modal');
        const nameElement = document.getElementById('modal-director-name');
        const contentElement = document.getElementById('modal-director-content');

        nameElement.textContent = director.director_name;

        contentElement.innerHTML = `
            <div style="margin-bottom: 20px;">
                <h3>Specialization</h3>
                <p>${director.specialization}</p>
            </div>
            <div style="margin-bottom: 20px;">
                <h3>Current Status</h3>
                <p>Status: <span class="director-status ${director.status}">${director.status}</span></p>
                <p>Current Load: ${director.current_load || 0} tasks</p>
                <p>Average Response Time: ${(director.average_response_time || 0).toFixed(1)}s</p>
                <p>Approval Rate: ${((director.approval_rate || 0) * 100).toFixed(1)}%</p>
            </div>
            ${director.last_recommendation ? `
            <div style="margin-bottom: 20px;">
                <h3>Latest Evaluation</h3>
                <p>Recommendation: ${director.last_recommendation}</p>
                <p>Confidence: ${((director.last_confidence || 0) * 100).toFixed(1)}%</p>
            </div>` : ''}
            ${director.last_error ? `
            <div style="margin-bottom: 20px;">
                <h3>Last Error</h3>
                <p style="color: #f44336;">${director.last_error}</p>
            </div>` : ''}
        `;

        modal.style.display = 'block';
    }

    showNotification(message, type = 'info') {
        const notification = document.getElementById('notification');
        notification.textContent = message;
        notification.className = `notification ${type} show`;

        setTimeout(() => {
            notification.className = `notification ${type}`;
        }, 3000);
    }

    setupEventListeners() {
        // Feedback type change handler
        document.getElementById('feedback-type').addEventListener('change', (e) => {
            const overrideGroup = document.getElementById('override-recommendation-group');
            if (['approve', 'reject', 'modify'].includes(e.target.value)) {
                overrideGroup.style.display = 'block';
            } else {
                overrideGroup.style.display = 'none';
            }
        });

        // Ping WebSocket periodically
        setInterval(() => {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000); // Ping every 30 seconds
    }

    startPeriodicUpdates() {
        // Update board status every 30 seconds
        setInterval(async () => {
            try {
                const response = await fetch(`${this.config.apiUrl}/status`);
                if (response.ok) {
                    const statusData = await response.json();
                    this.updateBoardStatus(statusData);
                }
            } catch (error) {
                console.error('Failed to update board status:', error);
            }
        }, 30000);

        // Update processing time for active tasks
        setInterval(() => {
            if (this.currentTaskStatus === 'in_progress') {
                const processingTimeElement = document.getElementById('processing-time');
                const currentTime = parseInt(processingTimeElement.textContent) + 1;
                processingTimeElement.textContent = `${currentTime}s`;
            }
        }, 1000);
    }
}

// Global functions for UI interactions
async function submitTask() {
    const description = document.getElementById('task-description').value.trim();
    const priority = document.getElementById('task-priority').value;

    if (!description) {
        dashboard.showNotification('Please enter a task description', 'warning');
        return;
    }

    try {
        const response = await fetch(`${dashboard.config.apiUrl}/task`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                task_description: description,
                user_id: 'user',
                priority: priority
            })
        });

        if (response.ok) {
            const result = await response.json();
            dashboard.showNotification('Task submitted successfully!', 'success');
            document.getElementById('task-description').value = '';
        } else {
            throw new Error('Failed to submit task');
        }
    } catch (error) {
        console.error('Error submitting task:', error);
        dashboard.showNotification('Failed to submit task', 'error');
    }
}

function requestModification() {
    showFeedbackModal('modify');
}

function approveRecommendation() {
    showFeedbackModal('approve');
}

function rejectRecommendation() {
    showFeedbackModal('reject');
}

function showFeedbackModal(type) {
    const modal = document.getElementById('feedback-modal');
    const typeSelect = document.getElementById('feedback-type');

    typeSelect.value = type;
    typeSelect.dispatchEvent(new Event('change'));

    modal.style.display = 'block';
}

function closeFeedbackModal() {
    const modal = document.getElementById('feedback-modal');
    modal.style.display = 'none';

    // Clear form
    document.getElementById('feedback-reasoning').value = '';
    document.getElementById('override-recommendation').value = '';
}

function closeDirectorModal() {
    const modal = document.getElementById('director-modal');
    modal.style.display = 'none';
}

async function submitFeedback() {
    if (!dashboard.currentTaskId) {
        dashboard.showNotification('No active task for feedback', 'warning');
        return;
    }

    const feedbackType = document.getElementById('feedback-type').value;
    const reasoning = document.getElementById('feedback-reasoning').value.trim();
    const overrideRecommendation = document.getElementById('override-recommendation').value.trim();

    if (!reasoning) {
        dashboard.showNotification('Please provide reasoning for your feedback', 'warning');
        return;
    }

    if (['approve', 'reject', 'modify'].includes(feedbackType) && !overrideRecommendation) {
        dashboard.showNotification('Please provide your recommendation', 'warning');
        return;
    }

    try {
        const response = await fetch(`${dashboard.config.apiUrl}/feedback/${dashboard.currentTaskId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                feedback_type: feedbackType,
                feedback_content: reasoning,
                override_recommendation: overrideRecommendation || null,
                reasoning: reasoning
            })
        });

        if (response.ok) {
            dashboard.showNotification('Feedback submitted successfully!', 'success');
            closeFeedbackModal();
        } else {
            throw new Error('Failed to submit feedback');
        }
    } catch (error) {
        console.error('Error submitting feedback:', error);
        dashboard.showNotification('Failed to submit feedback', 'error');
    }
}

// Close modals when clicking outside
window.onclick = function(event) {
    const directorModal = document.getElementById('director-modal');
    const feedbackModal = document.getElementById('feedback-modal');

    if (event.target === directorModal) {
        closeDirectorModal();
    }
    if (event.target === feedbackModal) {
        closeFeedbackModal();
    }
}

// Initialize dashboard when page loads
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new BoardRoomDashboard();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, can reduce update frequency
        console.log('Page hidden');
    } else {
        // Page is visible again, resume normal updates
        console.log('Page visible');
        if (dashboard && !dashboard.isConnected) {
            dashboard.connectWebSocket();
        }
    }
});

// Export for debugging
window.boardDashboard = () => dashboard;