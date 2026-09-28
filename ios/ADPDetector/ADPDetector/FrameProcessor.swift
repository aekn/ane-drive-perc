import AVFoundation
import CoreMedia
import Foundation

enum DroppedFrameReason {
    case late
    case outOfBuffers
    case discontinuity
    case unknown
}

struct CaptureStatistics: Equatable {
    private(set) var deliveredFrames: UInt64 = 0
    private(set) var droppedFrames: UInt64 = 0
    private(set) var lateFrames: UInt64 = 0
    private(set) var outOfBuffersFrames: UInt64 = 0
    private(set) var discontinuityFrames: UInt64 = 0
    private(set) var unknownDroppedFrames: UInt64 = 0

    mutating func recordDeliveredFrame() {
        deliveredFrames += 1
    }

    mutating func recordDroppedFrame(reason: DroppedFrameReason) {
        droppedFrames += 1
        switch reason {
        case .late:
            lateFrames += 1
        case .outOfBuffers:
            outOfBuffersFrames += 1
        case .discontinuity:
            discontinuityFrames += 1
        case .unknown:
            unknownDroppedFrames += 1
        }
    }
}

struct FrameTiming: Equatable {
    let captureTime: TimeInterval?
    let processingStartTime: TimeInterval
    let processingEndTime: TimeInterval

    var processingDuration: TimeInterval {
        max(0, processingEndTime - processingStartTime)
    }

    func age(at hostTime: TimeInterval) -> TimeInterval? {
        guard let captureTime else { return nil }
        return max(0, hostTime - captureTime)
    }
}

struct FrameResult {
    let tracks: [TrackSnapshot]
    let frameSize: CGSize
    let timing: FrameTiming
    let smoothedProcessingDuration: TimeInterval
    let captureStatistics: CaptureStatistics
}

final class FrameProcessor: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let detector: ANEDetector
    private let hostClock = CMClockGetHostTimeClock()
    private weak var session: AVCaptureSession?

    private let tracker = DetectionTracker()
    private var captureStatistics = CaptureStatistics()
    private var smoothedProcessingDuration: TimeInterval?
    private var needsWarmup = true
    private var reportedError = false

    var onResult: ((FrameResult) -> Void)?
    var onError: ((Error) -> Void)?

    init(detector: ANEDetector, session: AVCaptureSession) {
        self.detector = detector
        self.session = session
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        captureStatistics.recordDeliveredFrame()
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let captureTime = captureHostTime(for: sampleBuffer)
        let processingStartTime = currentHostTime()

        let detections: [Detection]
        do {
            detections = try detector.detect(sampleBuffer: sampleBuffer)
            reportedError = false
        } catch {
            if !reportedError {
                reportedError = true
                onError?(error)
            }
            return
        }

        let processingEndTime = currentHostTime()
        let timing = FrameTiming(
            captureTime: captureTime,
            processingStartTime: processingStartTime,
            processingEndTime: processingEndTime
        )

        if needsWarmup {
            needsWarmup = false
            tracker.reset()
            captureStatistics = CaptureStatistics()
            smoothedProcessingDuration = nil
            return
        }

        let duration = timing.processingDuration
        smoothedProcessingDuration = smoothedProcessingDuration.map {
            $0 + (duration - $0) * 0.15
        } ?? duration

        let result = FrameResult(
            tracks: tracker.update(
                detections: detections,
                at: captureTime ?? processingStartTime
            ),
            frameSize: CGSize(
                width: CVPixelBufferGetWidth(pixelBuffer),
                height: CVPixelBufferGetHeight(pixelBuffer)
            ),
            timing: timing,
            smoothedProcessingDuration: smoothedProcessingDuration ?? duration,
            captureStatistics: captureStatistics
        )
        onResult?(result)
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        captureStatistics.recordDroppedFrame(reason: droppedFrameReason(in: sampleBuffer))
    }

    func resetTracking() {
        tracker.reset()
    }

    private func currentHostTime() -> TimeInterval {
        CMTimeGetSeconds(CMClockGetTime(hostClock))
    }

    private func captureHostTime(for sampleBuffer: CMSampleBuffer) -> TimeInterval? {
        guard let synchronizationClock = session?.synchronizationClock else {
            return nil
        }

        let hostTime = CMSyncConvertTime(
            sampleBuffer.presentationTimeStamp,
            from: synchronizationClock,
            to: hostClock
        )
        let seconds = CMTimeGetSeconds(hostTime)
        return seconds.isFinite ? seconds : nil
    }

    private func droppedFrameReason(in sampleBuffer: CMSampleBuffer) -> DroppedFrameReason {
        guard let reason = CMGetAttachment(
            sampleBuffer,
            key: kCMSampleBufferAttachmentKey_DroppedFrameReason,
            attachmentModeOut: nil
        ) else {
            return .unknown
        }

        if CFEqual(reason, kCMSampleBufferDroppedFrameReason_FrameWasLate) {
            return .late
        }
        if CFEqual(reason, kCMSampleBufferDroppedFrameReason_OutOfBuffers) {
            return .outOfBuffers
        }
        if CFEqual(reason, kCMSampleBufferDroppedFrameReason_Discontinuity) {
            return .discontinuity
        }
        return .unknown
    }
}
