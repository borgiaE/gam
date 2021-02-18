import cv2
import depthai
import numpy as np

from pathlib import Path


def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def padded_point(point, padding, frame_shape=None):
    if frame_shape is None:
        return [
            point[0] - padding,
            point[1] - padding,
            point[0] + padding,
            point[1] + padding
        ]
    else:
        def norm(val, dim):
            return max(0, min(val, dim))
        if np.any(point - padding > frame_shape[:2]) or np.any(point + padding < 0):
            print(f"Unable to create padded box for point {point} with padding {padding} and frame shape {frame_shape[:2]}")
            return None

        return [
            norm(point[0] - padding, frame_shape[0]),
            norm(point[1] - padding, frame_shape[1]),
            norm(point[0] + padding, frame_shape[0]),
            norm(point[1] + padding, frame_shape[1])
        ]


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


def main():
    pipeline = depthai.Pipeline()

    print("Creating Color Camera...")
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(300, 300)
    cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(depthai.CameraBoardSocket.RGB)

    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("cam_out")
    cam.preview.link(cam_xout.input)

    print("Creating Face Detection Neural Network...")
    face_nn = pipeline.createNeuralNetwork()
    face_nn.setBlobPath(str(Path("./models/face-detection-retail-0004/face-detection-retail-0004.blob").resolve().absolute()))

    cam.preview.link(face_nn.input)

    face_nn_xout = pipeline.createXLinkOut()
    face_nn_xout.setStreamName("face_nn")
    face_nn.out.link(face_nn_xout.input)

    print("Creating Landmarks Detection Neural Network...")
    land_nn = pipeline.createNeuralNetwork()
    land_nn.setBlobPath(str(Path("./models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.blob").resolve().absolute()))

    land_nn_xin = pipeline.createXLinkIn()
    land_nn_xin.setStreamName("landmark_in")
    land_nn_xin.out.link(land_nn.input)
    land_nn_xout = pipeline.createXLinkOut()
    land_nn_xout.setStreamName("landmark_nn")
    land_nn.out.link(land_nn_xout.input)

    print("Creating GA Detection Neural Network...")
    fr_nn = pipeline.createNeuralNetwork()
    fr_nn.setBlobPath(str(Path("./models/r100-arcface/face_rec.blob").resolve().absolute()))

    fr_nn_xin = pipeline.createXLinkIn()
    fr_nn_xin.setStreamName("fr_in")
    fr_nn_xin.out.link(fr_nn.input)
    fr_nn_xout = pipeline.createXLinkOut()
    fr_nn_xout.setStreamName("fr_out")
    fr_nn.out.link(fr_nn_xout.input)

    device = depthai.Device(pipeline, "depthai-usb2.cmd")
    device.startPipeline()

    cam_out = device.getOutputQueue("cam_out")
    face_nn = device.getOutputQueue("face_nn")
    landmark_in = device.getInputQueue("landmark_in")
    landmark_nn = device.getOutputQueue(name="landmark_nn", maxSize=1, blocking=False)

    fr_nn_in = device.getInputQueue("fr_in")
    fr_nn_out = device.getOutputQueue(name="fr_out", maxSize=1, blocking=False)

    while True:
        n_frame = cam_out.tryGet()

        if n_frame is None:
            # print("No hay frame")
            continue

        new_frame = np.array(n_frame.getData()).reshape((3, 300, 300)).transpose(1, 2, 0).astype(np.uint8)
        debug_frame = np.ascontiguousarray(new_frame)

        try:
            bboxes_ = np.array(face_nn.get().getFirstLayerFp16())
        except RuntimeError as ex:
            print("No hay boxes")
            continue

        bboxes_ = bboxes_.reshape((bboxes_.size // 7, 7))
        bboxes = bboxes_[bboxes_[:, 2] > 0.7][:, 3:7]

        for raw_bbox in bboxes:
            bbox = frame_norm(new_frame, raw_bbox)
            det_frame = new_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            land_data = depthai.NNData()
            land_data.setLayer("0", to_planar(det_frame, (48, 48)))
            landmark_in.send(land_data)

            cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)

            try:
                land_in = landmark_nn.get().getFirstLayerFp16()
            except RuntimeError as ex:
                print("No hay putnos")
                continue

            left = bbox[0]
            top = bbox[1]

            land_data = frame_norm(det_frame, land_in)
            land_data[::2] += left
            land_data[1::2] += top

            left_bbox = padded_point(land_data[:2], padding=30, frame_shape=new_frame.shape)
            right_bbox = padded_point(land_data[2:4], padding=30, frame_shape=new_frame.shape)
            nose = land_data[4:6]

        cv2.imshow("Camera view", debug_frame)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
