import QuartzCore
import UIKit

struct OverlayDetection {
    let id: UInt64
    let rect: CGRect
    let classID: Int
    let label: String
    let score: Float
}

private let palette: [UIColor] = [
    UIColor(red: 0.0 / 255.0, green: 114.0 / 255.0, blue: 189.0 / 255.0, alpha: 1),
    UIColor(red: 217.0 / 255.0, green: 83.0 / 255.0, blue: 25.0 / 255.0, alpha: 1),
    UIColor(red: 237.0 / 255.0, green: 177.0 / 255.0, blue: 32.0 / 255.0, alpha: 1),
    UIColor(red: 126.0 / 255.0, green: 47.0 / 255.0, blue: 142.0 / 255.0, alpha: 1),
    UIColor(red: 119.0 / 255.0, green: 172.0 / 255.0, blue: 48.0 / 255.0, alpha: 1),
    UIColor(red: 77.0 / 255.0, green: 190.0 / 255.0, blue: 238.0 / 255.0, alpha: 1),
    UIColor(red: 162.0 / 255.0, green: 20.0 / 255.0, blue: 47.0 / 255.0, alpha: 1),
    UIColor(red: 76.0 / 255.0, green: 76.0 / 255.0, blue: 76.0 / 255.0, alpha: 1),
    UIColor(red: 153.0 / 255.0, green: 153.0 / 255.0, blue: 0.0 / 255.0, alpha: 1),
    UIColor(red: 255.0 / 255.0, green: 0.0 / 255.0, blue: 127.0 / 255.0, alpha: 1),
]

final class BoxOverlayView: UIView {
    private var trackLayers = [UInt64: TrackOverlayLayer]()
    private let hudLabel = UILabel()
    private var visibleObjectCount = 0
    private var hudText = ""

#if DEBUG
    private var processingMS = 0.0
    private var frameAgeMS: Double?
    private var droppedFrames: UInt64 = 0
#endif

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isOpaque = false
        isUserInteractionEnabled = false
        setupHUD()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        layoutHUD()
    }

#if DEBUG
    func updateDiagnostics(
        processingMS: Double,
        frameAgeMS: Double?,
        droppedFrames: UInt64
    ) {
        self.processingMS = processingMS
        self.frameAgeMS = frameAgeMS
        self.droppedFrames = droppedFrames
        updateHUD()
    }
#endif

    func updateDetections(_ detections: [OverlayDetection]) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)

        var visibleIDs = Set<UInt64>()
        visibleIDs.reserveCapacity(detections.count)

        let displayScale = max(traitCollection.displayScale, 1)
        for detection in detections {
            visibleIDs.insert(detection.id)
            let trackLayer = trackLayers[detection.id] ?? makeTrackLayer(
                id: detection.id
            )
            trackLayer.update(
                detection,
                in: bounds,
                scale: displayScale
            )
        }

        let staleIDs = trackLayers.keys.filter { !visibleIDs.contains($0) }
        for id in staleIDs {
            trackLayers.removeValue(forKey: id)?.removeFromSuperlayer()
        }

        CATransaction.commit()

        if visibleObjectCount != detections.count {
            visibleObjectCount = detections.count
            updateHUD()
        }
    }

    private func setupHUD() {
        hudLabel.font = .monospacedSystemFont(ofSize: 13, weight: .medium)
        hudLabel.textColor = .white
        hudLabel.backgroundColor = UIColor.black.withAlphaComponent(0.62)
        hudLabel.isUserInteractionEnabled = false
        hudLabel.layer.cornerRadius = 2
        hudLabel.layer.masksToBounds = true
        addSubview(hudLabel)
        updateHUD()
    }

    private func makeTrackLayer(id: UInt64) -> TrackOverlayLayer {
        let trackLayer = TrackOverlayLayer()
        layer.insertSublayer(trackLayer, below: hudLabel.layer)
        trackLayers[id] = trackLayer
        return trackLayer
    }

    private func updateHUD() {
#if DEBUG
        let processing = String(format: "%.1f ms", processingMS)
        let age = frameAgeMS.map { String(format: "%.1f ms", $0) } ?? "n/a"
        let text = "  ANE-S  ·  proc \(processing)  ·  age \(age)  ·  "
            + "drop \(droppedFrames)  ·  \(objectCountText)  "
#else
        let text = "  ANE-S  ·  \(objectCountText)  "
#endif
        guard text != hudText else { return }

        hudText = text
        hudLabel.text = text
        hudLabel.sizeToFit()
        layoutHUD()
    }

    private var objectCountText: String {
        visibleObjectCount == 1 ? "1 object" : "\(visibleObjectCount) objects"
    }

    private func layoutHUD() {
        let size = hudLabel.bounds.size
        hudLabel.frame = CGRect(
            x: safeAreaInsets.left + 12,
            y: safeAreaInsets.top + 10,
            width: size.width,
            height: max(size.height, 24)
        )
    }
}

private final class TrackOverlayLayer: CALayer {
    private let boxLayer = CALayer()
    private let tagLayer = CALayer()
    private let textLayer = CATextLayer()

    private var displayedText = ""
    private var displayedClassID = -1
    private var tagSize = CGSize.zero

    override init() {
        super.init()

        boxLayer.borderWidth = 2
        addSublayer(boxLayer)

        addSublayer(tagLayer)

        textLayer.alignmentMode = .left
        textLayer.isWrapped = false
        tagLayer.addSublayer(textLayer)
    }

    override init(layer: Any) {
        super.init(layer: layer)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func update(
        _ detection: OverlayDetection,
        in overlayBounds: CGRect,
        scale: CGFloat
    ) {
        updateContent(detection, scale: scale)
        updateGeometry(detection.rect, in: overlayBounds)
    }

    private func updateContent(_ detection: OverlayDetection, scale: CGFloat) {
        let color = palette[detection.classID % palette.count]
        if detection.classID != displayedClassID {
            displayedClassID = detection.classID
            boxLayer.borderColor = color.cgColor
            tagLayer.backgroundColor = color.withAlphaComponent(0.88).cgColor
        }

        let text = "\(detection.label) \(Int((detection.score * 100).rounded()))%"
        guard text != displayedText || textLayer.contentsScale != scale else {
            return
        }

        displayedText = text
        textLayer.contentsScale = scale

        let font = UIFont.systemFont(ofSize: 12, weight: .semibold)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: UIColor.white,
        ]
        let attributedText = NSAttributedString(
            string: text,
            attributes: attributes
        )
        let textSize = (text as NSString).size(withAttributes: attributes)

        textLayer.string = attributedText
        textLayer.frame = CGRect(
            x: 4,
            y: 3,
            width: ceil(textSize.width),
            height: ceil(textSize.height)
        )
        tagSize = CGSize(
            width: ceil(textSize.width) + 8,
            height: ceil(textSize.height) + 6
        )
    }

    private func updateGeometry(_ rect: CGRect, in overlayBounds: CGRect) {
        frame = overlayBounds
        let box = rect.intersection(overlayBounds)
        guard !box.isNull, box.width > 1, box.height > 1 else {
            isHidden = true
            return
        }

        isHidden = false
        boxLayer.frame = box

        let tagX = min(
            max(box.minX, overlayBounds.minX),
            max(overlayBounds.minX, overlayBounds.maxX - tagSize.width)
        )
        let preferredY = box.minY >= tagSize.height
            ? box.minY - tagSize.height
            : box.minY
        let tagY = min(
            max(preferredY, overlayBounds.minY),
            max(overlayBounds.minY, overlayBounds.maxY - tagSize.height)
        )
        tagLayer.frame = CGRect(
            x: tagX,
            y: tagY,
            width: tagSize.width,
            height: tagSize.height
        )
    }
}
