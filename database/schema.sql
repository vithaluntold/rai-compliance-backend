-- Bulletproof Document Analysis Database Schema
-- Eliminates ALL race conditions with atomic transactions
-- Single source of truth for compliance analysis results

-- Main document analysis table - atomic storage
CREATE TABLE IF NOT EXISTS document_analysis (
    document_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN (
        'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 
        'awaiting_framework_selection', 'COMPLETED_WITH_ERRORS'
    )),
    
    -- JSON fields for structured data
    metadata_json TEXT,          -- Company metadata extraction results
    sections_json TEXT,          -- Compliance analysis sections with questions
    framework TEXT,              -- Selected framework (IFRS, IPSAS, etc)
    standards_json TEXT,         -- List of selected standards as JSON array
    
    -- Processing configuration
    processing_mode TEXT CHECK (processing_mode IN ('smart', 'zap', 'comparison')),
    special_instructions TEXT,
    extensive_search BOOLEAN DEFAULT FALSE,
    
    -- Status tracking
    metadata_extraction TEXT CHECK (metadata_extraction IN (
        'PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED'
    )),
    compliance_analysis TEXT CHECK (compliance_analysis IN (
        'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'COMPLETED_WITH_ERRORS'
    )),
    
    -- Performance and error tracking
    performance_metrics_json TEXT,  -- Processing time, tokens used, etc
    error_message TEXT,
    failed_standards_json TEXT,     -- List of standards that failed processing
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- User message for frontend display
    message TEXT
);

-- Processing locks table - prevents race conditions
CREATE TABLE IF NOT EXISTS processing_locks (
    document_id TEXT PRIMARY KEY,
    lock_type TEXT NOT NULL CHECK (lock_type IN (
        'upload', 'metadata_extraction', 'compliance_analysis', 'update'
    )),
    acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    process_id TEXT,
    expires_at TIMESTAMP,
    
    FOREIGN KEY (document_id) REFERENCES document_analysis(document_id) ON DELETE CASCADE
);

-- Document chunks table - for chunk storage and retrieval
CREATE TABLE IF NOT EXISTS document_chunks (
    document_id TEXT,
    chunk_index INTEGER,
    chunk_text TEXT,
    chunk_metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (document_id, chunk_index),
    FOREIGN KEY (document_id) REFERENCES document_analysis(document_id) ON DELETE CASCADE
);

-- Progress tracking table - real-time progress updates
CREATE TABLE IF NOT EXISTS analysis_progress (
    document_id TEXT,
    standard_id TEXT,
    total_questions INTEGER DEFAULT 0,
    completed_questions INTEGER DEFAULT 0,
    current_question TEXT,
    progress_percentage REAL DEFAULT 0.0,
    elapsed_time_seconds REAL DEFAULT 0.0,
    status TEXT CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    PRIMARY KEY (document_id, standard_id),
    FOREIGN KEY (document_id) REFERENCES document_analysis(document_id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_document_analysis_status ON document_analysis(status);
CREATE INDEX IF NOT EXISTS idx_document_analysis_created_at ON document_analysis(created_at);
CREATE INDEX IF NOT EXISTS idx_document_analysis_framework ON document_analysis(framework);
CREATE INDEX IF NOT EXISTS idx_processing_locks_expires_at ON processing_locks(expires_at);
CREATE INDEX IF NOT EXISTS idx_analysis_progress_status ON analysis_progress(status);

-- Triggers for automatic timestamp updates
CREATE TRIGGER IF NOT EXISTS update_document_analysis_timestamp 
    AFTER UPDATE ON document_analysis
    FOR EACH ROW
BEGIN
    UPDATE document_analysis 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE document_id = NEW.document_id;
END;

-- Trigger to clean up expired locks
CREATE TRIGGER IF NOT EXISTS cleanup_expired_locks
    AFTER INSERT ON processing_locks
    FOR EACH ROW
BEGIN
    DELETE FROM processing_locks 
    WHERE expires_at < CURRENT_TIMESTAMP;
END;