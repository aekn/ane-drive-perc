import Foundation

struct TrackSnapshot: Equatable {
    let id: UInt64
    let bbox: BBox
    let classID: Int
    let label: String
    let score: Float
    let measurementTime: TimeInterval
    let expirationTime: TimeInterval

    private let centerVelocityX: Float
    private let centerVelocityY: Float
    private let widthVelocity: Float
    private let heightVelocity: Float
    private let predictionHorizon: TimeInterval

    init(
        id: UInt64,
        bbox: BBox,
        classID: Int,
        label: String,
        score: Float,
        measurementTime: TimeInterval,
        expirationTime: TimeInterval,
        centerVelocityX: Float,
        centerVelocityY: Float,
        widthVelocity: Float,
        heightVelocity: Float,
        predictionHorizon: TimeInterval
    ) {
        self.id = id
        self.bbox = bbox
        self.classID = classID
        self.label = label
        self.score = score
        self.measurementTime = measurementTime
        self.expirationTime = expirationTime
        self.centerVelocityX = centerVelocityX
        self.centerVelocityY = centerVelocityY
        self.widthVelocity = widthVelocity
        self.heightVelocity = heightVelocity
        self.predictionHorizon = predictionHorizon
    }

    func isVisible(at hostTime: TimeInterval) -> Bool {
        hostTime <= expirationTime
    }

    func predictedBBox(at hostTime: TimeInterval) -> BBox {
        let interval = min(
            max(0, hostTime - measurementTime),
            predictionHorizon
        )
        return bbox.predicted(
            centerVelocityX: centerVelocityX,
            centerVelocityY: centerVelocityY,
            widthVelocity: widthVelocity,
            heightVelocity: heightVelocity,
            after: interval
        )
    }
}

private struct BoxVelocity: Equatable {
    var centerX: Float = 0
    var centerY: Float = 0
    var width: Float = 0
    var height: Float = 0

    func interpolated(to other: BoxVelocity, fraction: Float) -> BoxVelocity {
        let t = min(max(fraction, 0), 1)
        return BoxVelocity(
            centerX: centerX + (other.centerX - centerX) * t,
            centerY: centerY + (other.centerY - centerY) * t,
            width: width + (other.width - width) * t,
            height: height + (other.height - height) * t
        )
    }
}

final class DetectionTracker {
    private struct Track {
        let id: UInt64
        let classID: Int
        let label: String
        var bbox: BBox
        var score: Float
        var velocity = BoxVelocity()
        var hasVelocity = false
        var measurementTime: TimeInterval
    }

    private struct Match {
        let track: Int
        let detection: Int
        let similarity: Float
    }

    private let minimumIoU: Float = 0.05
    private let maximumCenterDistance: Float = 1.25
    private let maximumAreaRatio: Float = 4.0
    private let minimumVelocityInterval: TimeInterval = 1.0 / 240.0
    private let velocityTimeConstant: TimeInterval = 0.12
    private let scoreTimeConstant: TimeInterval = 0.10
    private let staleInterval: TimeInterval = 0.30
    private let predictionHorizon: TimeInterval = 0.20

    private var tracks = [Track]()
    private var nextID: UInt64 = 0

    func update(
        detections: [Detection],
        at measurementTime: TimeInterval
    ) -> [TrackSnapshot] {
        let time = measurementTime
        tracks.removeAll {
            elapsed(from: $0.measurementTime, to: time) > staleInterval
        }

        let predictedBoxes = tracks.map {
            predictedBBox(for: $0, at: time)
        }
        var matches = [Match]()
        matches.reserveCapacity(tracks.count * detections.count)

        for trackIndex in tracks.indices {
            for detectionIndex in detections.indices {
                let track = tracks[trackIndex]
                let detection = detections[detectionIndex]
                guard track.classID == detection.classID else { continue }

                let predicted = predictedBoxes[trackIndex]
                guard let similarity = associationSimilarity(
                    predicted,
                    detection.bbox
                ) else {
                    continue
                }
                matches.append(
                    Match(
                        track: trackIndex,
                        detection: detectionIndex,
                        similarity: similarity
                    )
                )
            }
        }
        matches.sort { $0.similarity > $1.similarity }

        var usedTracks = Set<Int>()
        var usedDetections = Set<Int>()
        for match in matches {
            guard !usedTracks.contains(match.track),
                  !usedDetections.contains(match.detection)
            else {
                continue
            }

            usedTracks.insert(match.track)
            usedDetections.insert(match.detection)
            update(
                track: &tracks[match.track],
                with: detections[match.detection],
                at: time
            )
        }

        for detectionIndex in detections.indices
        where !usedDetections.contains(detectionIndex) {
            let detection = detections[detectionIndex]
            tracks.append(
                Track(
                    id: nextID,
                    classID: detection.classID,
                    label: detection.label,
                    bbox: detection.bbox,
                    score: detection.score,
                    measurementTime: time
                )
            )
            nextID += 1
        }

        return tracks
            .map(snapshot)
            .sorted { $0.score > $1.score }
    }

    func reset() {
        tracks.removeAll(keepingCapacity: true)
        nextID = 0
    }

    private func update(
        track: inout Track,
        with detection: Detection,
        at measurementTime: TimeInterval
    ) {
        let interval = elapsed(
            from: track.measurementTime,
            to: measurementTime
        )
        if interval >= minimumVelocityInterval {
            let measuredVelocity = velocity(
                from: track.bbox,
                to: detection.bbox,
                over: interval
            )
            if track.hasVelocity {
                let fraction = smoothingFraction(
                    interval: interval,
                    timeConstant: velocityTimeConstant
                )
                track.velocity = track.velocity.interpolated(
                    to: measuredVelocity,
                    fraction: fraction
                )
            } else {
                track.velocity = measuredVelocity
                track.hasVelocity = true
            }

            let scoreFraction = smoothingFraction(
                interval: interval,
                timeConstant: scoreTimeConstant
            )
            track.score += (detection.score - track.score) * scoreFraction
        } else {
            track.score = detection.score
        }

        track.bbox = detection.bbox
        track.measurementTime = measurementTime
    }

    private func predictedBBox(for track: Track, at hostTime: TimeInterval) -> BBox {
        guard track.hasVelocity else { return track.bbox }
        let interval = min(
            elapsed(from: track.measurementTime, to: hostTime),
            predictionHorizon
        )
        return track.bbox.predicted(
            centerVelocityX: track.velocity.centerX,
            centerVelocityY: track.velocity.centerY,
            widthVelocity: track.velocity.width,
            heightVelocity: track.velocity.height,
            after: interval
        )
    }

    private func associationSimilarity(
        _ predicted: BBox,
        _ detection: BBox
    ) -> Float? {
        let areaRatio = max(predicted.area, detection.area)
            / max(min(predicted.area, detection.area), .leastNonzeroMagnitude)
        guard areaRatio <= maximumAreaRatio else { return nil }

        let iou = predicted.iou(with: detection)
        let centerDistance = predicted.normalizedCenterDistance(to: detection)
        guard iou >= minimumIoU || centerDistance <= maximumCenterDistance else {
            return nil
        }

        let proximity = max(0, 1 - centerDistance / maximumCenterDistance)
        return iou * 0.75 + proximity * 0.25
    }

    private func velocity(
        from oldBox: BBox,
        to newBox: BBox,
        over interval: TimeInterval
    ) -> BoxVelocity {
        let scale = Float(1 / interval)
        return BoxVelocity(
            centerX: (newBox.centerX - oldBox.centerX) * scale,
            centerY: (newBox.centerY - oldBox.centerY) * scale,
            width: (newBox.width - oldBox.width) * scale,
            height: (newBox.height - oldBox.height) * scale
        )
    }

    private func smoothingFraction(
        interval: TimeInterval,
        timeConstant: TimeInterval
    ) -> Float {
        Float(1 - exp(-interval / timeConstant))
    }

    private func snapshot(_ track: Track) -> TrackSnapshot {
        TrackSnapshot(
            id: track.id,
            bbox: track.bbox,
            classID: track.classID,
            label: track.label,
            score: track.score,
            measurementTime: track.measurementTime,
            expirationTime: track.measurementTime + staleInterval,
            centerVelocityX: track.hasVelocity ? track.velocity.centerX : 0,
            centerVelocityY: track.hasVelocity ? track.velocity.centerY : 0,
            widthVelocity: track.hasVelocity ? track.velocity.width : 0,
            heightVelocity: track.hasVelocity ? track.velocity.height : 0,
            predictionHorizon: predictionHorizon
        )
    }

    private func elapsed(
        from start: TimeInterval,
        to end: TimeInterval
    ) -> TimeInterval {
        guard start.isFinite, end.isFinite else { return 0 }
        return max(0, end - start)
    }
}

private extension BBox {
    var centerX: Float { (x1 + x2) * 0.5 }
    var centerY: Float { (y1 + y2) * 0.5 }

    func predicted(
        centerVelocityX: Float,
        centerVelocityY: Float,
        widthVelocity: Float,
        heightVelocity: Float,
        after interval: TimeInterval
    ) -> BBox {
        let time = Float(interval)
        let predictedWidth = clamp(
            width + widthVelocity * time,
            minimum: width * 0.5,
            maximum: width * 1.5
        )
        let predictedHeight = clamp(
            height + heightVelocity * time,
            minimum: height * 0.5,
            maximum: height * 1.5
        )
        let predictedCenterX = centerX + centerVelocityX * time
        let predictedCenterY = centerY + centerVelocityY * time

        return BBox(
            x1: clamp(predictedCenterX - predictedWidth * 0.5),
            y1: clamp(predictedCenterY - predictedHeight * 0.5),
            x2: clamp(predictedCenterX + predictedWidth * 0.5),
            y2: clamp(predictedCenterY + predictedHeight * 0.5)
        )
    }

    func normalizedCenterDistance(to other: BBox) -> Float {
        let dx = centerX - other.centerX
        let dy = centerY - other.centerY
        let distance = hypotf(dx, dy)
        let scale = max(
            hypotf(width, height),
            hypotf(other.width, other.height),
            0.05
        )
        return distance / scale
    }

    private func clamp(_ value: Float) -> Float {
        min(max(value, 0), 1)
    }

    private func clamp(
        _ value: Float,
        minimum: Float,
        maximum: Float
    ) -> Float {
        min(max(value, minimum), maximum)
    }
}
