-- FlorNeuroPulse — Privacy-First Biomarker Telemetry Schema
-- Only non-reversible scalar metrics and dynamic anomaly scores are persisted.
-- Raw audio/video streams are processed transiently in-memory and never stored.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 1. User accounts (privacy-first: no biometric raw data stored)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(128),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    consent_given BOOLEAN DEFAULT FALSE,
    baseline_profile_version INTEGER DEFAULT 1
);

-- 2. 7-Day Dynamic Baselines (rolling scalar profiles per user)
CREATE TABLE baselines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    audio_f0_mean DOUBLE PRECISION,
    audio_f0_std DOUBLE PRECISION,
    audio_jitter_mean DOUBLE PRECISION,
    audio_shimmer_mean DOUBLE PRECISION,
    audio_hnr_mean DOUBLE PRECISION,
    vision_ear_mean DOUBLE PRECISION,
    vision_lip_asym_mean DOUBLE PRECISION,
    vision_head_pitch_mean DOUBLE PRECISION,
    vision_head_yaw_mean DOUBLE PRECISION,
    baseline_version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_baselines_user_window ON baselines(user_id, window_start DESC);

-- 3. Active Diagnostic Sessions (ephemeral session metadata only)
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(64) UNIQUE NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    status VARCHAR(32) DEFAULT 'active' CHECK (status IN ('active','completed','aborted')),
    device_id VARCHAR(128),
    consent_snapshot JSONB
);
CREATE INDEX idx_sessions_user_status ON sessions(user_id, status);

-- 4. Scalar Time-Series Biomarker Telemetry Logs (non-reversible metrics only)
CREATE TABLE telemetry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ DEFAULT NOW(),
    -- Audio 15D scalar metrics (MFCCs aggregated + f0/jitter/shimmer/HNR + ZCR + centroid)
    mfcc_1 DOUBLE PRECISION, mfcc_2 DOUBLE PRECISION, mfcc_3 DOUBLE PRECISION,
    mfcc_4 DOUBLE PRECISION, mfcc_5 DOUBLE PRECISION, mfcc_6 DOUBLE PRECISION,
    mfcc_7 DOUBLE PRECISION, mfcc_8 DOUBLE PRECISION, mfcc_9 DOUBLE PRECISION,
    mfcc_10 DOUBLE PRECISION, mfcc_11 DOUBLE PRECISION, mfcc_12 DOUBLE PRECISION,
    mfcc_13 DOUBLE PRECISION,
    spectral_centroid DOUBLE PRECISION,
    zcr DOUBLE PRECISION,
    f0_fundamental DOUBLE PRECISION,
    f0_jitter DOUBLE PRECISION,
    shimmer DOUBLE PRECISION,
    hnr DOUBLE PRECISION,
    -- Vision 5D scalar metrics (mesh-derived ratios / proxies)
    lip_corner_asymmetry DOUBLE PRECISION,
    vertical_lip_displacement DOUBLE PRECISION,
    eye_aspect_ratio DOUBLE PRECISION,
    head_pitch_proxy DOUBLE PRECISION,
    head_yaw_proxy DOUBLE PRECISION,
    -- Cross-modal inference output
    anomaly_index DOUBLE PRECISION CHECK (anomaly_index >= 0.0 AND anomaly_index <= 1.0),
    risk_tier VARCHAR(16) CHECK (risk_tier IN ('low','moderate','high','critical')),
    -- Metadata
    model_version VARCHAR(32),
    source_stream VARCHAR(32) CHECK (source_stream IN ('audio','vision','fused'))
);
CREATE INDEX idx_telemetry_session_time ON telemetry(session_id, recorded_at DESC);
CREATE INDEX idx_telemetry_user_time ON telemetry(user_id, recorded_at DESC);
CREATE INDEX idx_telemetry_anomaly ON telemetry(anomaly_index DESC) WHERE anomaly_index > 0.7;

-- 5. Audit / Purge Log (transient processing confirmation — no media retained)
CREATE TABLE processing_audit (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    event_type VARCHAR(32) CHECK (event_type IN ('stream_open','frame_processed','inference_run','stream_close','purge_confirmed')),
    event_timestamp TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    raw_media_persisted BOOLEAN DEFAULT FALSE
);
