import CoreGraphics

struct BBox: Equatable {
    var x1: Float
    var y1: Float
    var x2: Float
    var y2: Float

    var width: Float { max(0, x2 - x1) }
    var height: Float { max(0, y2 - y1) }
    var area: Float { width * height }

    func rect(in size: CGSize) -> CGRect {
        CGRect(
            x: CGFloat(x1) * size.width,
            y: CGFloat(y1) * size.height,
            width: CGFloat(width) * size.width,
            height: CGFloat(height) * size.height
        )
    }

    func iou(with other: BBox) -> Float {
        let ix1 = max(x1, other.x1)
        let iy1 = max(y1, other.y1)
        let ix2 = min(x2, other.x2)
        let iy2 = min(y2, other.y2)
        let intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        guard intersection > 0 else { return 0 }
        return intersection / max(area + other.area - intersection, .leastNonzeroMagnitude)
    }
}

struct Detection {
    let bbox: BBox
    let score: Float
    let classID: Int
    let label: String
}
