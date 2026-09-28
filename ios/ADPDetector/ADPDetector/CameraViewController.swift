import AVFoundation
import QuartzCore
import UIKit

final class CameraViewController: UIViewController {
    private static let modelName = "ane_s"

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let inferenceQueue = DispatchQueue(
        label: "camera.inference",
        qos: .userInitiated
    )

    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var overlayView: BoxOverlayView!
    private var frameProcessor: FrameProcessor?
    private var latestResult: FrameResult?
    private var displayLink: CADisplayLink?
    private var displayLinkTarget: DisplayLinkTarget?
    private var errorLabel: UILabel?
    private var sessionObservers = [NSObjectProtocol]()
    private var isSessionConfigured = false
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var previewRotationObservation: NSKeyValueObservation?
    private var captureRotationObservation: NSKeyValueObservation?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        setupPreview()
        setupOverlay()
        observeSession()
        loadFrameProcessor()
        requestCameraAccess()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        startDisplayLink()
    }

    override func viewWillDisappear(_ animated: Bool) {
        stopDisplayLink()
        super.viewWillDisappear(animated)
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        overlayView.frame = view.bounds
    }

    deinit {
        displayLink?.invalidate()
        let center = NotificationCenter.default
        sessionObservers.forEach { center.removeObserver($0) }
    }

    private func setupPreview() {
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
    }

    private func setupOverlay() {
        overlayView = BoxOverlayView(frame: view.bounds)
        view.addSubview(overlayView)
    }

    private func loadFrameProcessor() {
        guard let modelURL = Bundle.main.url(
            forResource: Self.modelName,
            withExtension: "mlmodelc"
        ) else {
            showError(CameraError.modelNotFound(Self.modelName).localizedDescription)
            return
        }

        ANEDetector.load(modelURL: modelURL) { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                switch result {
                case let .success(detector):
                    self.installFrameProcessor(detector: detector)
                case let .failure(error):
                    self.showError(
                        "Unable to load ANE-S.\n\(error.localizedDescription)"
                    )
                }
            }
        }
    }

    private func installFrameProcessor(detector: ANEDetector) {
        let processor = FrameProcessor(detector: detector, session: session)
        processor.onResult = { [weak self] result in
            DispatchQueue.main.async {
                self?.accept(result)
            }
        }
        processor.onError = { [weak self] error in
            DispatchQueue.main.async {
                self?.showError("Inference failed.\n\(error.localizedDescription)")
            }
        }

        frameProcessor = processor
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.videoOutput.setSampleBufferDelegate(
                processor,
                queue: self.inferenceQueue
            )
        }
    }

    private func requestCameraAccess() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStartSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self else { return }
                if granted {
                    self.configureAndStartSession()
                } else {
                    DispatchQueue.main.async {
                        self.showError(
                            "Camera access is required for the live demo."
                        )
                    }
                }
            }
        default:
            showError(
                "Camera access is disabled. Enable it in Settings to run the demo."
            )
        }
    }

    private func configureAndStartSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            do {
                if !self.isSessionConfigured {
                    let device = try self.configureSession()
                    self.isSessionConfigured = true
                    DispatchQueue.main.async {
                        self.configureRotation(for: device)
                    }
                }
                if !self.session.isRunning {
                    self.session.startRunning()
                }
            } catch {
                DispatchQueue.main.async {
                    self.showError(
                        "Camera setup failed.\n\(error.localizedDescription)"
                    )
                }
            }
        }
    }

    private func configureSession() throws -> AVCaptureDevice {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        if session.canSetSessionPreset(.hd1280x720) {
            session.sessionPreset = .hd1280x720
        }

        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        ) else {
            throw CameraError.cameraUnavailable
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw CameraError.cannotAddInput
        }
        session.addInput(input)

        let pixelFormat = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        if videoOutput.availableVideoPixelFormatTypes.contains(pixelFormat) {
            videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            ]
        }
        videoOutput.alwaysDiscardsLateVideoFrames = true

        guard session.canAddOutput(videoOutput) else {
            throw CameraError.cannotAddOutput
        }
        session.addOutput(videoOutput)
        return device
    }

    private func observeSession() {
        let center = NotificationCenter.default
        sessionObservers = [
            center.addObserver(
                forName: AVCaptureSession.wasInterruptedNotification,
                object: session,
                queue: .main
            ) { [weak self] _ in
                self?.sessionWasInterrupted()
            },
            center.addObserver(
                forName: AVCaptureSession.interruptionEndedNotification,
                object: session,
                queue: .main
            ) { [weak self] _ in
                self?.sessionInterruptionEnded()
            },
            center.addObserver(
                forName: AVCaptureSession.runtimeErrorNotification,
                object: session,
                queue: .main
            ) { [weak self] notification in
                self?.sessionRuntimeError(notification)
            },
        ]
    }

    private func sessionWasInterrupted() {
        clearTracks()
    }

    private func sessionInterruptionEnded() {
        restartSessionIfNeeded()
    }

    private func sessionRuntimeError(_ notification: Notification) {
        let error = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError
        if error?.domain == AVFoundationErrorDomain,
           error?.code == AVError.Code.mediaServicesWereReset.rawValue
        {
            restartSessionIfNeeded()
            return
        }

        showError(
            "Camera session failed.\n"
                + (error?.localizedDescription ?? "Unknown camera error.")
        )
    }

    private func restartSessionIfNeeded() {
        sessionQueue.async { [weak self] in
            guard let self,
                  self.isSessionConfigured,
                  !self.session.isRunning
            else {
                return
            }
            self.session.startRunning()
        }
    }

    private func configureRotation(for device: AVCaptureDevice) {
        let coordinator = AVCaptureDevice.RotationCoordinator(
            device: device,
            previewLayer: previewLayer
        )
        rotationCoordinator = coordinator

        previewRotationObservation = coordinator.observe(
            \.videoRotationAngleForHorizonLevelPreview,
            options: [.initial, .new]
        ) { [weak self] coordinator, _ in
            self?.applyRotation(
                coordinator.videoRotationAngleForHorizonLevelPreview,
                to: self?.previewLayer.connection
            )
        }

        captureRotationObservation = coordinator.observe(
            \.videoRotationAngleForHorizonLevelCapture,
            options: [.initial, .new]
        ) { [weak self] coordinator, _ in
            self?.applyRotation(
                coordinator.videoRotationAngleForHorizonLevelCapture,
                to: self?.videoOutput.connection(with: .video)
            )
        }
    }

    private func applyRotation(
        _ angle: CGFloat,
        to connection: AVCaptureConnection?
    ) {
        guard let connection,
              connection.isVideoRotationAngleSupported(angle)
        else {
            return
        }
        connection.videoRotationAngle = angle
    }

    private func accept(_ result: FrameResult) {
        clearError()
        latestResult = result
#if DEBUG
        let frameAgeMS = result.timing.age(at: CACurrentMediaTime()).map {
            $0 * 1000.0
        }
        overlayView.updateDiagnostics(
            processingMS: result.smoothedProcessingDuration * 1000.0,
            frameAgeMS: frameAgeMS,
            droppedFrames: result.captureStatistics.droppedFrames
        )
#endif
    }

    private func clearTracks() {
        latestResult = nil
        overlayView.updateDetections([])
        guard let frameProcessor else { return }
        inferenceQueue.async {
            frameProcessor.resetTracking()
        }
    }

    private func startDisplayLink() {
        guard displayLink == nil else { return }

        let target = DisplayLinkTarget(controller: self)
        let displayLink = CADisplayLink(
            target: target,
            selector: #selector(DisplayLinkTarget.displayLinkDidFire(_:))
        )
        displayLink.preferredFrameRateRange = CAFrameRateRange(
            minimum: 30,
            maximum: 60,
            preferred: 60
        )
        displayLink.add(to: .main, forMode: .common)

        displayLinkTarget = target
        self.displayLink = displayLink
    }

    private func stopDisplayLink() {
        displayLink?.invalidate()
        displayLink = nil
        displayLinkTarget = nil
    }

    fileprivate func displayLinkDidFire(_ displayLink: CADisplayLink) {
        guard let result = latestResult else {
            overlayView.updateDetections([])
            return
        }

        let displayTime = displayLink.targetTimestamp
        let detections = result.tracks.compactMap { track -> OverlayDetection? in
            guard track.isVisible(at: displayTime) else { return nil }

            let outputRect = track.predictedBBox(at: displayTime).rect(
                in: result.frameSize
            )
            let metadataRect = videoOutput.metadataOutputRectConverted(
                fromOutputRect: outputRect
            )
            let layerRect = previewLayer.layerRectConverted(
                fromMetadataOutputRect: metadataRect
            )
            return OverlayDetection(
                id: track.id,
                rect: layerRect,
                classID: track.classID,
                label: track.label,
                score: track.score
            )
        }
        overlayView.updateDetections(detections)
    }

    private func showError(_ message: String) {
        let label = errorLabel ?? makeErrorLabel()
        label.text = message
        label.isHidden = false
        errorLabel = label
    }

    private func clearError() {
        errorLabel?.isHidden = true
    }

    private func makeErrorLabel() -> UILabel {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.numberOfLines = 0
        label.textAlignment = .center
        label.textColor = .white
        label.font = .systemFont(ofSize: 17, weight: .medium)
        view.addSubview(label)
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 32),
            label.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -32),
            label.centerYAnchor.constraint(equalTo: view.centerYAnchor),
        ])
        return label
    }
}

private final class DisplayLinkTarget: NSObject {
    private weak var controller: CameraViewController?

    init(controller: CameraViewController) {
        self.controller = controller
    }

    @objc func displayLinkDidFire(_ displayLink: CADisplayLink) {
        controller?.displayLinkDidFire(displayLink)
    }
}

enum CameraError: LocalizedError {
    case modelNotFound(String)
    case cameraUnavailable
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case let .modelNotFound(name):
            "Core ML model \(name) is not bundled with the app."
        case .cameraUnavailable:
            "The back camera is unavailable."
        case .cannotAddInput:
            "The camera input could not be added to the capture session."
        case .cannotAddOutput:
            "The video output could not be added to the capture session."
        }
    }
}
