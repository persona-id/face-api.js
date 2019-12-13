import 'core-js/features/string/ends-with';
import 'core-js/features/object/assign';
import * as tf from '@tensorflow/tfjs-core';
import { draw as drawBase } from 'tfjs-image-recognition-base';
import * as drawExtended from './draw';
export { tf };
export * from 'tfjs-image-recognition-base';
export * from './ageGenderNet/index';
declare const draw: {
    drawContour(ctx: CanvasRenderingContext2D, points: import("tfjs-image-recognition-base").Point[], isClosed?: boolean): void;
    drawDetections(canvasArg: string | HTMLCanvasElement, detections: import("./classes").FaceDetection | import("tfjs-image-recognition-base").IRect | {
        detection: import("./classes").FaceDetection;
    } | import("tfjs-image-recognition-base").IBoundingBox | drawExtended.TDrawDetectionsInput[]): void;
    drawFaceExpressions(canvasArg: string | HTMLCanvasElement, faceExpressions: import("./faceExpressionNet").FaceExpressions | {
        expressions: import("./faceExpressionNet").FaceExpressions;
    } | drawExtended.DrawFaceExpressionsInput[], minConfidence?: number, textFieldAnchor?: import("tfjs-image-recognition-base").IPoint | undefined): void;
    drawFaceLandmarks(canvasArg: string | HTMLCanvasElement, faceLandmarks: import("./classes").FaceLandmarks | import("./factories").WithFaceLandmarks<{
        detection: import("./classes").FaceDetection;
    }, import("./classes").FaceLandmarks68> | drawExtended.DrawFaceLandmarksInput[]): void;
    DrawFaceLandmarksOptions: typeof drawExtended.DrawFaceLandmarksOptions;
    DrawFaceLandmarks: typeof drawExtended.DrawFaceLandmarks;
    DrawBoxOptions: typeof drawBase.DrawBoxOptions;
    DrawBox: typeof drawBase.DrawBox;
    AnchorPosition: typeof drawBase.AnchorPosition;
    DrawTextFieldOptions: typeof drawBase.DrawTextFieldOptions;
    DrawTextField: typeof drawBase.DrawTextField;
};
export { draw };
export * from './classes/index';
export * from './dom/index';
export * from './faceExpressionNet/index';
export * from './faceLandmarkNet/index';
export * from './faceRecognitionNet/index';
export * from './factories/index';
export * from './globalApi/index';
export * from './mtcnn/index';
export * from './ssdMobilenetv1/index';
export * from './tinyFaceDetector/index';
export * from './tinyYolov2/index';
export * from './euclideanDistance';
export * from './resizeResults';
