import cv2
import sys
import numpy as np
import depthai as dai
import time
import blobconverter

from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

'''
This example attaches a NeuralNetwork node directly to the SPI output. The corresponding ESP32 example shows how to decode it.

Make sure you have something to handle the SPI protocol on the other end! See the included ESP32 example.
'''
print("Creating SPI pipeline: ")
print("COLOR CAM -> DetectionNetwork -> SPI OUT")

labelMap = [
    "background",
	"aeroplane",
	"bicycle",
	"bird",
	"boat",
	"bottle",
	"bus",
	"car",
	"cat",
	"chair",
	"cow",
	"diningtable",
	"dog",
	"horse",
	"motorbike",
	"person",
	"pottedplant",
	"sheep",
	"sofa",
	"train",
	"tvmonitor"
]

pipeline = dai.Pipeline()

# set up NN node
nn1 = pipeline.create(dai.node.MobileNetDetectionNetwork)
nn1.setBlobPath(str(blobconverter.from_zoo('mobilenet-ssd', shaves=6, version='2021.4')))

# set up color camera and link to NN node
colorCam = pipeline.create(dai.node.ColorCamera)

# XLinkOut node for RGB image out
previewOut = pipeline.create(dai.node.XLinkOut)
previewOut.setStreamName("preview")

# XLinkOut node for nn detections metadata
nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nn")

colorCam.setPreviewSize(300, 300)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
colorCam.preview.link(nn1.input)

# set up SPI out node and link to nn1
spiOut = pipeline.create(dai.node.SPIOut)
spiOut.setStreamName("spimetaout")
spiOut.setBusId(0)
spiOut.input.setBlocking(False)
spiOut.input.setQueueSize(2)
nn1.out.link(spiOut.input)

colorCam.preview.link(previewOut.input)
nn1.out.link(nnOut.input)

def intersection(a, b):
    x1 = max(min(a.xmin, a.xmax), min(b.xmin, b.xmax))
    y1 = max(min(a.ymin, a.ymax), min(b.ymin, b.ymax))
    x2 = min(max(a.xmin, a.xmax), max(b.xmin, b.xmax))
    y2 = min(max(a.ymin, a.ymax), max(b.ymin, b.ymax))
    if x1<x2 and y1<y2:
        return Rectangle(x1, y1, x2, y2)
    else:
        return None

def overlapRatio(a, b):
    """Returns a ration of how much Rectangle BV in contained in/overlaps with Rectangle A. 0 is returned if there is no overlap. 1 is returned if B is fully inside A."""
    # returns None if rectangles don't intersect
    if a.xmax <= b.xmin or b.xmax <= a.xmin or a.ymin >= b.ymax or b.ymin >= a.ymax:
        return 0

    if a.xmax >= b.xmax and a.xmin <= b.xmin:
        # X: B fully inside A
        width = b.xmax - b.xmin
    elif b.xmax >= a.xmax and b.xmin <= a.xmin:
        # X: A fully inside B
        width = a.xmax - a.xmin
    elif a.xmax >= b.xmax:
        # X: B overlaps left edge of A
        width = b.xmax - a.xmin
    else:
        # X: B overlaps right edge of A
        width = a.xmax - b.xmin

    if a.ymin <= b.ymin and a.ymax >= b.ymax:
        # Y: B fully inside A
        height = b.ymax - b.ymin
    elif b.ymin <= a.ymin and b.ymax >= a.ymax:
        # Y: A fully inside B
        height = a.ymax - a.ymin
    elif a.ymin <= b.ymin:
        # Y: B overlaps bottom edge of A
        height = a.ymax - b.ymin
    else:
        # Y: B overlaps top edge of A
        height = b.ymax - a.ymin

    return (width * height) / ((b.xmax - b.xmin) * (b.ymax - b.ymin))

MIN_OVERLAP = 0.6

with dai.Device(pipeline) as device:
    previewQueue = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []

    slots = [
        Rectangle(0, 0, 0.5, 1),
        Rectangle(0.5, 0, 1, 1)
    ]

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        overlay = frame.copy()  # For drawing semi-transparent overlaps
        for detection in detections:
            # Ignore if not one of the "approved" categories
            if not detection.label in [7]:
                continue

            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            # Coords (normalized)
            # cv2.putText(frame, f"{bbox[0]:.4f}, {bbox[1]:.4f} -> {bbox[2]:.4f}, {bbox[3]:.4f}", (bbox[0] + 10, bbox[3] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,255))
            
            # Raw coords
            # cv2.putText(frame, f"{detection.xmin:.4f}, {detection.ymin:.4f} -> {detection.xmax:.4f}, {detection.ymax:.4f}", (bbox[0] + 10, bbox[3] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,255))
            
            # Overlaps
            for indx, slot in enumerate(slots):
                overlap = overlapRatio(slot, Rectangle(detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                occupied = overlapRatio(Rectangle(detection.xmin, detection.ymin, detection.xmax, detection.ymax), slot)

                # Ignore if not in this slot
                if overlap < MIN_OVERLAP:
                    continue

                cv2.putText(frame, f"Slot {indx}: {overlap * 100:.0f}%/{occupied * 100:.0f}%", ((10 + 150 * indx), frame.shape[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 1)

                fill = intersection(slot, Rectangle(detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                fillBox = frameNorm(overlay, (fill.xmin, fill.ymin, fill.xmax, fill.ymax))
                if intersection is not None:
                    cv2.rectangle(overlay, (fillBox[0], fillBox[1]), (fillBox[2], fillBox[3]), (0,255,255), -1)

        # Following line overlays transparent rectangle over the image
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw slots
        for slot in slots:
            slotFrame = frameNorm(frame, slot)
            cv2.rectangle(frame, (slotFrame[0], slotFrame[1]), (slotFrame[2], slotFrame[3]), (0, 255, 255), 2)

        # Show the frame
        cv2.imshow(name, frame)

    while True:
        inRgb = previewQueue.tryGet()
        inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()

        if inDet is not None:
            detections = inDet.detections

        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            print("Exiting in 3 seconds...")
            time.sleep(3)
            break
