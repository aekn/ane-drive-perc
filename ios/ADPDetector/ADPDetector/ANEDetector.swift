import AVFoundation
import CoreML
import Foundation
import Vision

final class ANEDetector {
    private struct Candidate {
        let query: Int
        let classID: Int
        let score: Float
    }

    private static let classes = [
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
    ]

    private let request: VNCoreMLRequest
    private let numQueries = 300
    private let numClasses = ANEDetector.classes.count
    private let numTopQueries = 300
    private let maxDetections = 80
    private let scoreThreshold: Float = 0.45
    private let nmsIoUThreshold: Float = 0.60

    static func load(
        modelURL: URL,
        completion: @escaping (Result<ANEDetector, Error>) -> Void
    ) {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine

        if #available(iOS 18.0, *) {
            var hints = MLOptimizationHints()
            hints.specializationStrategy = .fastPrediction
            configuration.optimizationHints = hints
        }

        MLModel.load(contentsOf: modelURL, configuration: configuration) { result in
            completion(
                result.flatMap { model in
                    Result {
                        try ANEDetector(model: model)
                    }
                }
            )
        }
    }

    private init(model: MLModel) throws {
        let visionModel = try VNCoreMLModel(for: model)
        request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFill
    }

    func detect(sampleBuffer: CMSampleBuffer) throws -> [Detection] {
        let handler = VNImageRequestHandler(
            cmSampleBuffer: sampleBuffer,
            orientation: .up
        )
        try handler.perform([request])

        guard let observations = request.results as? [VNCoreMLFeatureValueObservation] else {
            throw DetectorError.missingOutputs
        }

        var logits: MLMultiArray?
        var boxes: MLMultiArray?
        for observation in observations {
            switch observation.featureName {
            case "pred_logits":
                logits = observation.featureValue.multiArrayValue
            case "pred_boxes":
                boxes = observation.featureValue.multiArrayValue
            default:
                continue
            }
        }

        guard let logits, let boxes else {
            throw DetectorError.missingOutputs
        }
        return try postprocess(logits: logits, boxes: boxes)
    }

    private func postprocess(
        logits: MLMultiArray,
        boxes: MLMultiArray
    ) throws -> [Detection] {
        try validate(logits: logits, boxes: boxes)

        var candidates = [Candidate]()
        candidates.reserveCapacity(numQueries * numClasses)
        for query in 0 ..< numQueries {
            for classID in 0 ..< numClasses {
                candidates.append(
                    Candidate(
                        query: query,
                        classID: classID,
                        score: sigmoid(value(logits, 0, query, classID))
                    )
                )
            }
        }
        candidates.sort { $0.score > $1.score }

        var detections = [Detection]()
        detections.reserveCapacity(maxDetections)
        for candidate in candidates.prefix(numTopQueries) {
            guard candidate.score >= scoreThreshold else { break }

            let cx = value(boxes, 0, candidate.query, 0)
            let cy = value(boxes, 0, candidate.query, 1)
            let width = value(boxes, 0, candidate.query, 2)
            let height = value(boxes, 0, candidate.query, 3)
            let bbox = BBox(
                x1: clamp(cx - width / 2),
                y1: clamp(cy - height / 2),
                x2: clamp(cx + width / 2),
                y2: clamp(cy + height / 2)
            )
            guard bbox.width > 0, bbox.height > 0 else { continue }

            if detections.contains(where: {
                $0.classID == candidate.classID
                    && $0.bbox.iou(with: bbox) > nmsIoUThreshold
            }) {
                continue
            }

            detections.append(
                Detection(
                    bbox: bbox,
                    score: candidate.score,
                    classID: candidate.classID,
                    label: Self.classes[candidate.classID]
                )
            )
            if detections.count == maxDetections {
                break
            }
        }
        return detections
    }

    private func validate(logits: MLMultiArray, boxes: MLMultiArray) throws {
        guard supports(logits.dataType), supports(boxes.dataType) else {
            throw DetectorError.unsupportedOutputType
        }
        guard logits.shape.map(\.intValue) == [1, numQueries, numClasses] else {
            throw DetectorError.invalidOutputShape("pred_logits", logits.shape)
        }
        guard boxes.shape.map(\.intValue) == [1, numQueries, 4] else {
            throw DetectorError.invalidOutputShape("pred_boxes", boxes.shape)
        }
    }

    private func value(
        _ array: MLMultiArray,
        _ i0: Int,
        _ i1: Int,
        _ i2: Int
    ) -> Float {
        let offset = i0 * array.strides[0].intValue
            + i1 * array.strides[1].intValue
            + i2 * array.strides[2].intValue
        switch array.dataType {
        case .float16:
            return Float(
                array.dataPointer.assumingMemoryBound(to: Float16.self)[offset]
            )
        case .float32:
            return array.dataPointer.assumingMemoryBound(to: Float32.self)[offset]
        default:
            preconditionFailure("unsupported Core ML output type")
        }
    }

    private func supports(_ dataType: MLMultiArrayDataType) -> Bool {
        dataType == .float16 || dataType == .float32
    }

    private func sigmoid(_ value: Float) -> Float {
        1 / (1 + expf(-value))
    }

    private func clamp(_ value: Float) -> Float {
        min(max(value, 0), 1)
    }
}

enum DetectorError: LocalizedError {
    case missingOutputs
    case unsupportedOutputType
    case invalidOutputShape(String, [NSNumber])

    var errorDescription: String? {
        switch self {
        case .missingOutputs:
            "The model did not return pred_logits and pred_boxes."
        case .unsupportedOutputType:
            "The model outputs must use Float16 or Float32 tensors."
        case let .invalidOutputShape(name, shape):
            "Unexpected \(name) shape: \(shape)."
        }
    }
}
