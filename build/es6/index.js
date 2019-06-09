import * as tslib_1 from "tslib";
// Polyfill for IE support
import 'core-js/features/string/ends-with';
import * as tf from '@tensorflow/tfjs-core';
import { draw as drawBase } from 'tfjs-image-recognition-base';
import * as drawExtended from './draw';
export { tf };
export * from 'tfjs-image-recognition-base';
export * from './ageGenderNet/index';
var draw = tslib_1.__assign({}, drawBase, drawExtended);
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
//# sourceMappingURL=index.js.map