import Foundation
import Testing
@testable import ADPDetector

struct BBoxTests {
    @Test func intersectionOverUnion() {
        let a = BBox(x1: 0, y1: 0, x2: 1, y2: 1)
        let b = BBox(x1: 0.5, y1: 0.5, x2: 1, y2: 1)

        #expect(abs(a.iou(with: b) - 0.25) < 1e-6)
    }
}

struct FrameTimingTests {
    @Test func reportsProcessingDurationAndFrameAge() {
        let timing = FrameTiming(
            captureTime: 10,
            processingStartTime: 10.05,
            processingEndTime: 10.13
        )

        #expect(abs(timing.processingDuration - 0.08) < 1e-9)
        #expect(abs((timing.age(at: 10.15) ?? 0) - 0.15) < 1e-9)
    }

    @Test func missingCaptureTimeHasNoFrameAge() {
        let timing = FrameTiming(
            captureTime: nil,
            processingStartTime: 10,
            processingEndTime: 10.08
        )

        #expect(timing.age(at: 10.1) == nil)
    }

    @Test func durationsDoNotBecomeNegative() {
        let timing = FrameTiming(
            captureTime: 10,
            processingStartTime: 10.2,
            processingEndTime: 10.1
        )

        #expect(timing.processingDuration == 0)
        #expect(timing.age(at: 9.9) == 0)
    }
}

struct CaptureStatisticsTests {
    @Test func countsDeliveredAndDroppedFramesByReason() {
        var statistics = CaptureStatistics()

        statistics.recordDeliveredFrame()
        statistics.recordDeliveredFrame()
        statistics.recordDroppedFrame(reason: .late)
        statistics.recordDroppedFrame(reason: .outOfBuffers)
        statistics.recordDroppedFrame(reason: .discontinuity)
        statistics.recordDroppedFrame(reason: .unknown)

        #expect(statistics.deliveredFrames == 2)
        #expect(statistics.droppedFrames == 4)
        #expect(statistics.lateFrames == 1)
        #expect(statistics.outOfBuffersFrames == 1)
        #expect(statistics.discontinuityFrames == 1)
        #expect(statistics.unknownDroppedFrames == 1)
    }
}

struct DetectionTrackerTests {
    @Test func newDetectionsAreVisibleImmediately() {
        let tracker = DetectionTracker()
        let detection = car(
            BBox(x1: 0.20, y1: 0.20, x2: 0.40, y2: 0.40)
        )

        let tracks = tracker.update(detections: [detection], at: 10)

        #expect(tracks.count == 1)
        #expect(tracks[0].id == 0)
        #expect(tracks[0].bbox == detection.bbox)
        #expect(tracks[0].measurementTime == 10)
        #expect(tracks[0].isVisible(at: 10.29))
        #expect(!tracks[0].isVisible(at: 10.31))
    }

    @Test func matchingIsClassAware() {
        let tracker = DetectionTracker()
        let box = BBox(x1: 0.2, y1: 0.2, x2: 0.6, y2: 0.6)

        _ = tracker.update(detections: [car(box)], at: 10)
        let tracks = tracker.update(detections: [truck(box)], at: 10.08)

        #expect(tracks.count == 2)
        #expect(Set(tracks.map(\.classID)) == Set([2, 3]))
    }

    @Test func predictionUsesMeasuredVelocity() {
        let tracker = DetectionTracker()
        let first = car(BBox(x1: 0.10, y1: 0.20, x2: 0.30, y2: 0.40))
        let second = car(BBox(x1: 0.18, y1: 0.20, x2: 0.38, y2: 0.40))

        _ = tracker.update(detections: [first], at: 10)
        let tracks = tracker.update(detections: [second], at: 10.08)
        let predicted = tracks[0].predictedBBox(at: 10.12)

        #expect(abs(predicted.x1 - 0.22) < 1e-5)
        #expect(abs(predicted.x2 - 0.42) < 1e-5)
    }

    @Test func predictionHorizonIsBounded() {
        let tracker = DetectionTracker()
        let first = car(BBox(x1: 0.10, y1: 0.20, x2: 0.30, y2: 0.40))
        let second = car(BBox(x1: 0.18, y1: 0.20, x2: 0.38, y2: 0.40))

        _ = tracker.update(detections: [first], at: 10)
        let tracks = tracker.update(detections: [second], at: 10.08)
        let predicted = tracks[0].predictedBBox(at: 11)

        #expect(abs(predicted.x1 - 0.38) < 1e-5)
        #expect(abs(predicted.x2 - 0.58) < 1e-5)
    }

    @Test func predictedMotionSupportsLowOverlapAssociation() {
        let tracker = DetectionTracker()
        let first = car(BBox(x1: 0.10, y1: 0.20, x2: 0.20, y2: 0.30))
        let second = car(BBox(x1: 0.16, y1: 0.20, x2: 0.26, y2: 0.30))
        let third = car(BBox(x1: 0.22, y1: 0.20, x2: 0.32, y2: 0.30))

        let firstTrack = tracker.update(detections: [first], at: 10)[0]
        _ = tracker.update(detections: [second], at: 10.08)
        let thirdTrack = tracker.update(detections: [third], at: 10.16)[0]

        #expect(thirdTrack.id == firstTrack.id)
        #expect(thirdTrack.bbox == third.bbox)
    }

    @Test func staleTracksExpireByElapsedTime() {
        let tracker = DetectionTracker()
        let detection = car(BBox(x1: 0.2, y1: 0.2, x2: 0.4, y2: 0.4))

        _ = tracker.update(detections: [detection], at: 10)
        #expect(tracker.update(detections: [], at: 10.08).count == 1)
        #expect(tracker.update(detections: [], at: 10.31).isEmpty)
    }

    @Test func resetClearsTracksAndRestartsIdentifiers() {
        let tracker = DetectionTracker()
        let detection = car(BBox(x1: 0.2, y1: 0.2, x2: 0.4, y2: 0.4))

        _ = tracker.update(detections: [detection], at: 10)
        tracker.reset()
        let tracks = tracker.update(detections: [detection], at: 20)

        #expect(tracks.count == 1)
        #expect(tracks[0].id == 0)
    }

    private func car(_ bbox: BBox, score: Float = 0.9) -> Detection {
        Detection(bbox: bbox, score: score, classID: 2, label: "car")
    }

    private func truck(_ bbox: BBox, score: Float = 0.9) -> Detection {
        Detection(bbox: bbox, score: score, classID: 3, label: "truck")
    }
}
