import React, {useRef, useEffect} from "react";
import * as posenet from "@tensorflow-models/posenet";
import {drawBoundingBox, drawKeypoints, drawSkeleton} from "./utils/posenet";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import {styled} from "styled-components";

const App = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    useEffect(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");

        let checkRvalue = [];
        let checkHSSCValue = [];
        let checkOverlapValue = [];
        let prevHeadPosX = 0.0;
        let prevHeadPosY = 0.0;

        let isFall = false;

        function sendFallDetectionToServer() {
            const url = 'http://ec2-43-201-9-166.ap-northeast-2.compute.amazonaws.com:8080/alarm/push'

            fetch(url)
                .then(response => response.text())
                .then(data => {
                    console.log(data)
                })
                .catch(error => {
                    console.error("Error: ", error)
                })
        }

        const loadModelAndPredict = async () => {
            const model = await posenet.load({
                architecture: "MobileNetV1",
            });

            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;

            video.onloadeddata = () => {
                predict(model);
            };
        };

        const gru_predict = async (checkRvalue, checkHSSCValue, checkOverlapValue) => {
            const gruModel = await tf.loadLayersModel("/model.json");

            // let rList = checkRvalue.split(",");
            let rList = checkRvalue;
            let newRList = [];
            if (rList.length == 33) {
                rList = rList.slice(1);
                rList = rList.slice(0, rList.length - 2);
            } else {
                rList = rList.slice(1);
                rList = rList.slice(0, rList.length - 1);
            }
            let i = rList.indexOf("[null");
            delete rList[i];
            delete rList[i - 1];

            for (var rl in rList) {
                var real = parseFloat(rList[rl]).toFixed(3);
                newRList.push(real);
            }

            // let hsscList = checkHSSCValue.split(",");
            let hsscList = checkHSSCValue;
            let newHsscList = [];

            if (hsscList.length == 32) {
                hsscList = hsscList.slice(1);
                hsscList = hsscList.slice(0, hsscList.length - 1);
            } else {
                hsscList = hsscList.slice(1);
            }
            i = hsscList.indexOf("[null");
            delete hsscList[i];
            delete hsscList[i - 1];

            for (var hl in hsscList) {
                var real = parseFloat(hsscList[hl]).toFixed(3);
                newHsscList.push(real);
            }

            // let overlapList = checkOverlapValue.split(",");
            let overlapList = checkOverlapValue;
            let newOverlapList = [];

            if (overlapList.length == 33) {
                overlapList = overlapList.slice(1);
                overlapList = overlapList.slice(0, overlapList.length - 2);
            } else {
                overlapList = overlapList.slice(1);
                overlapList = overlapList.slice(0, overlapList.length - 1);
            }

            i = overlapList.indexOf("[null");
            delete overlapList[i];
            delete overlapList[i - 1];

            for (var ol in overlapList) {
                var real = parseFloat(overlapList[ol]).toFixed(3);
                newOverlapList.push(real);
            }

            let merged = newRList.concat(newHsscList);
            merged = merged.concat(newOverlapList);

            let x = tf.tensor(merged, [1, 30, 3], "float32");

            const prediction = gruModel.predict(x);
            const predictionArray = prediction.dataSync();
            const filteredPredictionArray = predictionArray.filter(item => item !== 1);
            const maxValue = Math.max(...filteredPredictionArray);

            if (0.5 < maxValue && !isFall) {
                console.log("Fall detected!");
                sendFallDetectionToServer();

                isFall = true;
            } else if (maxValue <= 0.5 && isFall) {
                // console.log("Fall not detected!");
                isFall = false;
            }
        };

        const predict = async (model) => {
            if (checkOverlapValue.length % 32 === 0) {
                console.log(checkRvalue);
                gru_predict(
                    JSON.stringify(checkRvalue),
                    JSON.stringify(checkHSSCValue),
                    JSON.stringify(checkOverlapValue)
                );
                checkRvalue = [];
                checkHSSCValue = [];
                checkOverlapValue = [];
            }

            const pose = await model.estimateSinglePose(video);
            canvas.width = video.width;
            canvas.height = video.height;

            context.clearRect(0, 0, canvas.width, canvas.height);

            drawKeypoints(pose.keypoints, 0.1, context);
            drawSkeleton(pose.keypoints, 0.1, context);

            let totalHeadX = 0.0;
            let totalHeadY = 0.0;

            let head = [];
            for (let i = 0; i < 7; i++) {
                head.push(pose.keypoints[i]);
                totalHeadX += pose.keypoints[i].position.x;
                totalHeadY += pose.keypoints[i].position.y;
            }
            let body = [];
            for (let i = 5; i < 13; i++) {
                if (i == 7 || i == 8 || i == 9 || i == 10) continue;
                body.push(pose.keypoints[i]);
            }
            let leg = [];
            for (let i = 11; i < 17; i++) {
                leg.push(pose.keypoints[i]);
            }

            const headBoundingBox = posenet.getBoundingBox(head);
            const bodyBoundingBox = posenet.getBoundingBox(body);
            const headArea =
                (headBoundingBox.maxX - headBoundingBox.minX) *
                (headBoundingBox.maxY - headBoundingBox.minY);
            if (headBoundingBox.maxY > bodyBoundingBox.minY) {
                const overlapWidth = Math.max(
                    0,
                    Math.min(
                        headBoundingBox.minX +
                        (headBoundingBox.maxX - headBoundingBox.minX),
                        bodyBoundingBox.minX + (bodyBoundingBox.maxX - bodyBoundingBox.minX)
                    ) - Math.max(headBoundingBox.minX, bodyBoundingBox.minX)
                );
                const overlapHeight = Math.max(
                    0,
                    Math.min(
                        headBoundingBox.minY +
                        (headBoundingBox.maxY - headBoundingBox.minY),
                        bodyBoundingBox.minY + (bodyBoundingBox.maxY - bodyBoundingBox.minY)
                    ) - Math.max(headBoundingBox.minY, bodyBoundingBox.minY)
                );
                const overlapArea = overlapWidth * overlapHeight;
                checkOverlapValue.push(overlapArea);
            } else {
                const overlapArea = 0.0;
                checkOverlapValue.push(overlapArea);
            }

            const legBoundingbox = posenet.getBoundingBox(leg);

            if (headBoundingBox.minY < bodyBoundingBox.maxY + 10) {
                console.log("head&body segment overlapped!");
            }

            if (bodyBoundingBox.minY < legBoundingbox.maxY + 10) {
                console.log("body&leg segment overlapped!");
            }

            drawBoundingBox(head, context);
            drawBoundingBox(body, context);
            drawBoundingBox(leg, context);

            if (prevHeadPosX == 0 && prevHeadPosY == 0) {
                prevHeadPosX = totalHeadX / 7;
                prevHeadPosY = totalHeadY / 7;
            } else {
                let posneg = 1;
                if (prevHeadPosY < totalHeadY / 7) posneg = -1;
                const distance = Math.sqrt(
                    Math.pow(totalHeadX / 7 - prevHeadPosX, 2) +
                    Math.pow(totalHeadY / 7 - prevHeadPosY, 2)
                );
                const speed = posneg * (distance / 33);
                checkHSSCValue.push(speed);
                prevHeadPosX = totalHeadX / 7;
                prevHeadPosY = totalHeadY / 7;
            }

            const boundingBox = posenet.getBoundingBox(pose.keypoints);
            checkRvalue.push(
                (boundingBox.maxX - boundingBox.minX) /
                (boundingBox.maxY - boundingBox.minY).toFixed(2)
            );

            gru_predict(checkRvalue, checkHSSCValue, checkOverlapValue);

            requestAnimationFrame(() => predict(model));
        };

        loadModelAndPredict();
    }, []);

    return (
        <Container>
            <Video
                ref={videoRef}
                width="800"
                height="600"
                muted
                autoPlay
                playsInline
            ></Video>
            <Canvas ref={canvasRef}></Canvas>
        </Container>
    );
};

export default App;

const Container = styled.div``;

const Video = styled.video`
  position: absolute;
  transform: translate(-50%, -50%);
  top: 50%;
  left: 50%;
`;

const Canvas = styled.canvas`
  position: absolute;
  transform: translate(-50%, -50%);
  top: 50%;
  left: 50%;
`;

